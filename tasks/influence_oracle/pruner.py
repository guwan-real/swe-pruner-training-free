from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from tf_pruning.protocol import LineScore, PruningRequest, PruningResult
from tf_pruning.selection import render_pruned_text, select_line_numbers
from tf_pruning.text import (
    TextBlock,
    build_query,
    code_aware_blocks,
    error_anchor_lines,
    structural_anchor_lines,
)


class LogLikelihoodScorer(Protocol):
    """Pluggable scorer for a fixed reference action."""

    name: str

    def log_likelihood(self, context: str, continuation: str) -> float:
        """Return log p(continuation | context); larger is better."""


class InfluenceObjective(Protocol):
    """Defines what behavior a deletion must preserve."""

    name: str

    def target(self, request: PruningRequest) -> str:
        """Return the reference continuation."""

    def prompt(self, request: PruningRequest, observation: str) -> str:
        """Build the causal-LM prefix for an observation variant."""


@dataclass(frozen=True)
class NextActionObjective:
    """Preserve the recorded coding-agent next action."""

    metadata_key: str = "next_action"
    instruction: str = (
        "Predict the coding agent's recorded next action from the available tool observation."
    )
    name: str = "next_action_log_likelihood"

    def target(self, request: PruningRequest) -> str:
        raw = request.metadata.get(self.metadata_key)
        if raw is None:
            raise ValueError(
                f"request metadata must contain {self.metadata_key!r} for the influence oracle"
            )
        target = str(raw)
        if not target:
            raise ValueError("next_action target must be non-empty")
        return target

    def prompt(self, request: PruningRequest, observation: str) -> str:
        query = build_query(
            request.query,
            path=request.path,
            recent_context=request.recent_context,
        )
        parts = [
            self.instruction,
            f"Tool type: {request.tool_type}",
        ]
        if query:
            parts.append(f"Goal and recent context: {query}")
        parts.extend(("Tool observation:", observation, "Next action:"))
        return "\n".join(parts) + "\n"


@dataclass(frozen=True)
class InfluenceOracleConfig:
    """Controls the deliberately expensive offline oracle."""

    strategy: str = "leave_one_out"
    block_max_lines: int = 16
    max_initial_blocks: int | None = 64
    max_evaluations: int | None = 2048
    protect_structure: bool = False
    protect_errors: bool = False
    protect_metadata_anchors: bool = False
    show_line_numbers: bool = True

    def __post_init__(self) -> None:
        if self.strategy not in {"leave_one_out", "hierarchical_greedy"}:
            raise ValueError("strategy must be 'leave_one_out' or 'hierarchical_greedy'")
        if self.block_max_lines < 1:
            raise ValueError("block_max_lines must be positive")
        if self.max_initial_blocks is not None and self.max_initial_blocks < 1:
            raise ValueError("max_initial_blocks must be positive")
        if self.max_evaluations is not None and self.max_evaluations < 1:
            raise ValueError("max_evaluations must be positive")


class HFLogLikelihoodScorer:
    """Reference-action likelihood from a local HF causal language model."""

    name = "hf-local-causal-lm"

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "auto",
        dtype: str = "auto",
        max_length: int | None = None,
        normalize: bool = False,
        trust_remote_code: bool = False,
        local_files_only: bool = True,
    ) -> None:
        if not model_path:
            raise ValueError("model_path must be non-empty")
        if not local_files_only:
            raise ValueError("influence_oracle only permits local_files_only=true")
        if max_length is not None and max_length < 2:
            raise ValueError("max_length must be at least 2")
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.max_length = max_length
        self.normalize = normalize
        self.trust_remote_code = trust_remote_code
        self.local_files_only = True
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._resolved_device: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "HF scoring requires the 'model' extra: pip install -e '.[model]'"
            ) from exc

        resolved_device = self.device
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        model_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.dtype != "auto":
            try:
                model_kwargs["torch_dtype"] = dtype_map[self.dtype.lower()]
            except KeyError as exc:
                raise ValueError(f"unsupported dtype: {self.dtype}") from exc

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=self.trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **model_kwargs,
        )
        model.eval()
        model.requires_grad_(False)
        model.to(resolved_device)

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._resolved_device = resolved_device

    def _effective_max_length(self) -> int:
        assert self._tokenizer is not None
        tokenizer_limit = getattr(self._tokenizer, "model_max_length", None)
        if isinstance(tokenizer_limit, int) and 2 <= tokenizer_limit < 1_000_000:
            return min(self.max_length, tokenizer_limit) if self.max_length else tokenizer_limit
        return self.max_length or 4096

    def log_likelihood(self, context: str, continuation: str) -> float:
        self._load()
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._model is not None

        torch = self._torch
        tokenizer = self._tokenizer
        prefix_ids = tokenizer.encode(context, add_special_tokens=False)
        target_ids = tokenizer.encode(continuation, add_special_tokens=False)
        if not target_ids:
            raise ValueError("reference continuation tokenized to zero tokens")

        max_length = self._effective_max_length()
        if len(target_ids) >= max_length:
            raise ValueError(
                f"reference action has {len(target_ids)} tokens but max_length "
                f"is {max_length}; increase max_length"
            )
        prefix_ids = prefix_ids[-(max_length - len(target_ids)) :]
        if not prefix_ids:
            start_id = tokenizer.bos_token_id
            if start_id is None:
                start_id = tokenizer.eos_token_id
            if start_id is None:
                raise RuntimeError("tokenizer needs a BOS or EOS token for empty context")
            prefix_ids = [int(start_id)]

        input_ids = prefix_ids + target_ids
        tensor = torch.tensor(
            [input_ids],
            dtype=torch.long,
            device=self._resolved_device,
        )
        attention_mask = torch.ones_like(tensor)
        with torch.inference_mode():
            logits = self._model(
                input_ids=tensor,
                attention_mask=attention_mask,
            ).logits

        target_start = len(prefix_ids)
        target_tensor = tensor[0, target_start:]
        prediction_logits = logits[
            0,
            target_start - 1 : len(input_ids) - 1,
            :,
        ]
        losses = torch.nn.functional.cross_entropy(
            prediction_logits.float(),
            target_tensor,
            reduction="none",
        )
        nll = losses.mean() if self.normalize else losses.sum()
        return -float(nll.item())


def _split_range(
    lines: Sequence[str],
    start: int,
    end: int,
    max_lines: int,
    *,
    kind: str,
) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    for chunk_start in range(start, end + 1, max_lines):
        chunk_end = min(end, chunk_start + max_lines - 1)
        blocks.append(
            TextBlock(
                start_line=chunk_start,
                end_line=chunk_end,
                text="\n".join(lines[chunk_start - 1 : chunk_end]),
                kind=kind,
            )
        )
    return blocks


def _partition_blocks(text: str, max_lines: int) -> list[TextBlock]:
    """Build a disjoint, complete partition from shared code-aware blocks."""

    lines = text.splitlines()
    if not lines:
        return []
    candidates = sorted(
        code_aware_blocks(text),
        key=lambda block: (
            block.start_line,
            -(block.end_line - block.start_line),
            block.end_line,
        ),
    )

    blocks: list[TextBlock] = []
    cursor = 1
    for block in candidates:
        start = max(1, block.start_line)
        end = min(len(lines), block.end_line)
        if start < cursor or end < start:
            continue
        if cursor < start:
            blocks.extend(
                _split_range(
                    lines,
                    cursor,
                    start - 1,
                    max_lines,
                    kind="gap",
                )
            )
        blocks.extend(
            _split_range(
                lines,
                start,
                end,
                max_lines,
                kind=block.kind,
            )
        )
        cursor = end + 1
    if cursor <= len(lines):
        blocks.extend(
            _split_range(
                lines,
                cursor,
                len(lines),
                max_lines,
                kind="gap",
            )
        )
    return blocks


def _metadata_anchor_lines(metadata: Mapping[str, Any]) -> set[int]:
    raw = metadata.get("anchor_lines", ())
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        return set()
    anchors: set[int] = set()
    for item in raw:
        try:
            anchors.add(int(item))
        except (TypeError, ValueError):
            continue
    return anchors


class _EvaluationLimit:
    def __init__(self, maximum: int | None) -> None:
        self.maximum = maximum
        self.count = 0

    def reserve(self, amount: int = 1) -> None:
        if self.maximum is not None and self.count + amount > self.maximum:
            raise RuntimeError(
                "influence oracle max_evaluations would be exceeded "
                f"({self.count + amount}>{self.maximum}); raise the limit or "
                "use fewer/larger blocks"
            )
        self.count += amount


class InfluenceOraclePruner:
    """Offline likelihood-damage oracle over observation deletions."""

    name = "influence_oracle"

    def __init__(
        self,
        scorer: LogLikelihoodScorer | None,
        objective: InfluenceObjective | None = None,
        config: InfluenceOracleConfig | None = None,
    ) -> None:
        self.scorer = scorer
        self.objective = objective or NextActionObjective()
        self.config = config or InfluenceOracleConfig()

    @staticmethod
    def _checked_likelihood(value: float) -> float:
        likelihood = float(value)
        if not math.isfinite(likelihood):
            raise ValueError(f"scorer returned a non-finite log likelihood: {likelihood}")
        return likelihood

    def _anchors(self, request: PruningRequest) -> set[int]:
        anchors: set[int] = set()
        if self.config.protect_structure:
            anchors.update(structural_anchor_lines(request.lines))
        if self.config.protect_errors:
            anchors.update(error_anchor_lines(request.lines))
        if self.config.protect_metadata_anchors:
            anchors.update(_metadata_anchor_lines(request.metadata))
        return {line_no for line_no in anchors if 1 <= line_no <= len(request.lines)}

    def _observation(
        self,
        lines: Sequence[str],
        kept: Sequence[int] | set[int],
    ) -> str:
        return render_pruned_text(
            lines,
            kept,
            show_line_numbers=self.config.show_line_numbers,
        )

    def _evaluate(
        self,
        request: PruningRequest,
        target_action: str,
        lines: Sequence[str],
        kept: set[int],
        limit: _EvaluationLimit,
        cache: dict[tuple[int, ...], float],
    ) -> float:
        key = tuple(sorted(kept))
        cached = cache.get(key)
        if cached is not None:
            return cached
        limit.reserve()
        observation = self._observation(lines, key)
        prompt = self.objective.prompt(request, observation)
        assert self.scorer is not None
        value = self._checked_likelihood(self.scorer.log_likelihood(prompt, target_action))
        cache[key] = value
        return value

    def _leave_one_out(
        self,
        request: PruningRequest,
        target_action: str,
        blocks: Sequence[TextBlock],
        anchors: set[int],
        limit: _EvaluationLimit,
    ) -> tuple[tuple[int, ...], list[float], list[tuple[str, ...]], float]:
        lines = request.lines
        all_lines = set(range(1, len(lines) + 1))
        cache: dict[tuple[int, ...], float] = {}
        full_likelihood = self._evaluate(
            request,
            target_action,
            lines,
            all_lines,
            limit,
            cache,
        )
        scores = [0.0] * len(lines)
        reasons: list[tuple[str, ...]] = [() for _ in lines]

        for block in blocks:
            removed = set(block.line_numbers)
            ablated_likelihood = self._evaluate(
                request,
                target_action,
                lines,
                all_lines - removed,
                limit,
                cache,
            )
            harm = full_likelihood - ablated_likelihood
            for line_no in block.line_numbers:
                scores[line_no - 1] = harm
                reason = [f"leave-one-block-out:{block.start_line}-{block.end_line}"]
                if line_no in anchors:
                    reason.append("protected-anchor")
                reasons[line_no - 1] = tuple(reason)

        kept = select_line_numbers(
            scores,
            request.budget,
            mandatory=anchors,
        )
        return kept, scores, reasons, full_likelihood

    @staticmethod
    def _line_units(
        lines: Sequence[str],
        active: set[int],
    ) -> list[TextBlock]:
        return [
            TextBlock(
                start_line=line_no,
                end_line=line_no,
                text=lines[line_no - 1],
                kind="line",
            )
            for line_no in sorted(active)
        ]

    def _hierarchical_greedy(
        self,
        request: PruningRequest,
        target_action: str,
        blocks: Sequence[TextBlock],
        anchors: set[int],
        limit: _EvaluationLimit,
    ) -> tuple[tuple[int, ...], list[float], list[tuple[str, ...]], float]:
        lines = request.lines
        target_lines = request.budget.target_lines(len(lines))
        if len(anchors) > target_lines:
            raise ValueError(
                "protected anchors exceed the line budget; disable protection "
                "or increase the budget"
            )

        active = set(range(1, len(lines) + 1))
        cache: dict[tuple[int, ...], float] = {}
        current_likelihood = self._evaluate(
            request,
            target_action,
            lines,
            active,
            limit,
            cache,
        )
        full_likelihood = current_likelihood
        units = list(blocks)
        line_scores = [0.0] * len(lines)
        reason_lists: list[list[str]] = [[] for _ in lines]
        refined = False
        step = 0

        while len(active) > target_lines:
            candidates = [
                unit
                for unit in units
                if set(unit.line_numbers).issubset(active)
                and not anchors.intersection(unit.line_numbers)
                and len(active) - len(unit.line_numbers) >= target_lines
            ]
            if not candidates:
                if refined:
                    raise RuntimeError(
                        "no legal greedy deletion remains under the requested "
                        "budget and anchor constraints"
                    )
                units = self._line_units(lines, active)
                refined = True
                continue

            limit.reserve(len(candidates))
            # Calls below are already reserved as one unbiased candidate batch.
            limit.count -= len(candidates)
            evaluated: list[tuple[float, int, int, TextBlock, float]] = []
            for unit in candidates:
                candidate_active = active - set(unit.line_numbers)
                candidate_likelihood = self._evaluate(
                    request,
                    target_action,
                    lines,
                    candidate_active,
                    limit,
                    cache,
                )
                harm = current_likelihood - candidate_likelihood
                evaluated.append(
                    (
                        harm,
                        unit.start_line,
                        unit.end_line,
                        unit,
                        candidate_likelihood,
                    )
                )
                for line_no in unit.line_numbers:
                    line_scores[line_no - 1] = harm

            harm, _, _, chosen, chosen_likelihood = min(
                evaluated,
                key=lambda item: (item[0], item[1], item[2]),
            )
            step += 1
            for line_no in chosen.line_numbers:
                reason_lists[line_no - 1].append(f"greedy-removed-step-{step}:harm={harm:.8g}")
            active.difference_update(chosen.line_numbers)
            current_likelihood = chosen_likelihood
            units = [unit for unit in units if unit is not chosen]

        for line_no in sorted(active):
            reason_lists[line_no - 1].append("greedy-kept")
            if line_no in anchors:
                reason_lists[line_no - 1].append("protected-anchor")
        return (
            tuple(sorted(active)),
            line_scores,
            [tuple(items) for items in reason_lists],
            full_likelihood,
        )

    def prune(self, request: PruningRequest) -> PruningResult:
        started = time.perf_counter()
        lines = request.lines
        line_count = len(lines)
        target_lines = request.budget.target_lines(line_count)
        if target_lines >= line_count:
            kept = tuple(range(1, line_count + 1))
            return PruningResult(
                method=self.name,
                original_line_count=line_count,
                kept_line_numbers=kept,
                pruned_text=self._observation(lines, kept),
                line_scores=tuple(
                    LineScore(line_no=index, score=0.0, reasons=("no-prune",)) for index in kept
                ),
                latency_ms=(time.perf_counter() - started) * 1000.0,
                metadata={
                    "strategy": self.config.strategy,
                    "target_lines": target_lines,
                    "evaluations": 0,
                    "model_forward_count": 0,
                },
                request_id=request.request_id,
            )
        if self.scorer is None:
            raise RuntimeError(
                "influence_oracle needs a scorer; set model_path in config "
                "or inject a LogLikelihoodScorer"
            )

        blocks = _partition_blocks(request.text, self.config.block_max_lines)
        if (
            self.config.max_initial_blocks is not None
            and len(blocks) > self.config.max_initial_blocks
        ):
            raise RuntimeError(
                f"observation produced {len(blocks)} blocks, exceeding "
                f"max_initial_blocks={self.config.max_initial_blocks}; use "
                "larger blocks or explicitly raise the offline-oracle limit"
            )
        anchors = self._anchors(request)
        target_action = self.objective.target(request)
        limit = _EvaluationLimit(self.config.max_evaluations)

        if self.config.strategy == "leave_one_out":
            kept, scores, reasons, full_likelihood = self._leave_one_out(
                request,
                target_action,
                blocks,
                anchors,
                limit,
            )
        else:
            kept, scores, reasons, full_likelihood = self._hierarchical_greedy(
                request,
                target_action,
                blocks,
                anchors,
                limit,
            )

        return PruningResult(
            method=self.name,
            original_line_count=line_count,
            kept_line_numbers=kept,
            pruned_text=self._observation(lines, kept),
            line_scores=tuple(
                LineScore(
                    line_no=line_no,
                    score=scores[line_no - 1],
                    reasons=reasons[line_no - 1],
                )
                for line_no in range(1, line_count + 1)
            ),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            metadata={
                "strategy": self.config.strategy,
                "target_lines": target_lines,
                "block_count": len(blocks),
                "evaluations": limit.count,
                "model_forward_count": limit.count,
                "full_log_likelihood": full_likelihood,
                "objective": getattr(
                    self.objective,
                    "name",
                    type(self.objective).__name__,
                ),
                "scorer": getattr(self.scorer, "name", type(self.scorer).__name__),
                "anchor_lines": sorted(anchors),
                "oracle_scope": "small-sample-offline",
            },
            request_id=request.request_id,
        )


__all__ = [
    "HFLogLikelihoodScorer",
    "InfluenceObjective",
    "InfluenceOracleConfig",
    "InfluenceOraclePruner",
    "LogLikelihoodScorer",
    "NextActionObjective",
]
