from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from tf_pruning.protocol import (
    LineScore,
    PruningRequest,
    PruningResult,
)
from tf_pruning.selection import render_pruned_text, select_line_numbers
from tf_pruning.text import (
    TextBlock,
    build_query,
    code_aware_blocks,
    error_anchor_lines,
    structural_anchor_lines,
)


class ConditionalSurprisalScorer(Protocol):
    """Pluggable conditional negative-log-likelihood scorer."""

    name: str

    def score(
        self,
        context: str,
        continuation: str,
        *,
        first_token_only: bool = False,
    ) -> float:
        """Return conditional surprisal in nats; larger means less expected."""


@dataclass(frozen=True)
class ConditionalPPLConfig:
    """Configuration for the coarse-to-fine ranking procedure."""

    coarse_top_fraction: float = 0.5
    coarse_top_blocks: int | None = None
    block_max_lines: int = 24
    fine_context_lines: int = 2
    coarse_weight: float = 0.35
    fine_weight: float = 0.65
    coarse_first_token_only: bool = False
    first_token_only: bool = True
    protect_structure: bool = True
    protect_errors: bool = True
    protect_metadata_anchors: bool = True
    expand_anchor_context: bool = False
    show_line_numbers: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.coarse_top_fraction <= 1.0:
            raise ValueError("coarse_top_fraction must be in (0, 1]")
        if self.coarse_top_blocks is not None and self.coarse_top_blocks < 1:
            raise ValueError("coarse_top_blocks must be positive")
        if self.block_max_lines < 1:
            raise ValueError("block_max_lines must be positive")
        if self.fine_context_lines < 0:
            raise ValueError("fine_context_lines must be non-negative")
        if self.coarse_weight < 0.0 or self.fine_weight < 0.0:
            raise ValueError("score weights must be non-negative")
        if self.coarse_weight + self.fine_weight <= 0.0:
            raise ValueError("at least one score weight must be positive")


class HFConditionalSurprisalScorer:
    """Conditional surprisal from a local Hugging Face causal language model.

    Imports and model loading are delayed until the first ``score`` call.
    ``local_files_only`` is intentionally fixed to true so an experiment cannot
    unexpectedly contact the Hub from an offline/server run.
    """

    name = "hf-local-causal-lm"

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "auto",
        dtype: str = "auto",
        max_length: int | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool = True,
    ) -> None:
        if not model_path:
            raise ValueError("model_path must be non-empty")
        if not local_files_only:
            raise ValueError("conditional_ppl only permits local_files_only=true")
        if max_length is not None and max_length < 2:
            raise ValueError("max_length must be at least 2")
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.max_length = max_length
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
        configured = self.max_length
        tokenizer_limit = getattr(self._tokenizer, "model_max_length", None)
        if isinstance(tokenizer_limit, int) and 2 <= tokenizer_limit < 1_000_000:
            return min(configured, tokenizer_limit) if configured else tokenizer_limit
        return configured or 4096

    def score(
        self,
        context: str,
        continuation: str,
        *,
        first_token_only: bool = False,
    ) -> float:
        self._load()
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._model is not None

        torch = self._torch
        tokenizer = self._tokenizer
        prefix_ids = tokenizer.encode(context, add_special_tokens=False)
        target_ids = tokenizer.encode(continuation, add_special_tokens=False)
        if not target_ids:
            return 0.0
        if first_token_only:
            target_ids = target_ids[:1]

        max_length = self._effective_max_length()
        if len(target_ids) >= max_length:
            target_ids = target_ids[: max_length - 1]
        prefix_budget = max_length - len(target_ids)
        prefix_ids = prefix_ids[-prefix_budget:]

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
        return float(losses.mean().item())


def _split_block(block: TextBlock, max_lines: int) -> list[TextBlock]:
    if len(block.line_numbers) <= max_lines:
        return [block]
    lines = block.text.splitlines()
    pieces: list[TextBlock] = []
    for offset in range(0, len(lines), max_lines):
        piece_lines = lines[offset : offset + max_lines]
        start = block.start_line + offset
        end = start + len(piece_lines) - 1
        pieces.append(
            TextBlock(
                start_line=start,
                end_line=end,
                text="\n".join(piece_lines),
                kind=block.kind,
            )
        )
    return pieces


def _candidate_blocks(text: str, max_lines: int) -> list[TextBlock]:
    """Return code-aware candidates plus chunks for uncovered lines."""

    lines = text.splitlines()
    if not lines:
        return []
    candidates: list[TextBlock] = []
    for block in code_aware_blocks(text):
        candidates.extend(_split_block(block, max_lines))

    covered = {
        line_no
        for block in candidates
        for line_no in block.line_numbers
        if 1 <= line_no <= len(lines)
    }
    line_no = 1
    while line_no <= len(lines):
        if line_no in covered:
            line_no += 1
            continue
        start = line_no
        while line_no <= len(lines) and line_no not in covered and line_no - start < max_lines:
            line_no += 1
        end = line_no - 1
        candidates.append(
            TextBlock(
                start_line=start,
                end_line=end,
                text="\n".join(lines[start - 1 : end]),
                kind="gap",
            )
        )

    unique = {
        (block.start_line, block.end_line, block.text, block.kind): block for block in candidates
    }
    return sorted(
        unique.values(),
        key=lambda block: (block.start_line, block.end_line, block.kind),
    )


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


class ConditionalPPLPruner:
    """Rank surprising blocks, refine the best blocks at line granularity."""

    name = "conditional_ppl"

    def __init__(
        self,
        scorer: ConditionalSurprisalScorer | None,
        config: ConditionalPPLConfig | None = None,
    ) -> None:
        self.scorer = scorer
        self.config = config or ConditionalPPLConfig()

    @staticmethod
    def _checked_score(value: float) -> float:
        score = float(value)
        if not math.isfinite(score):
            raise ValueError(f"scorer returned a non-finite value: {score}")
        return score

    def _base_context(self, request: PruningRequest) -> str:
        query = build_query(
            request.query,
            path=request.path,
            recent_context=request.recent_context,
        )
        parts = [
            "Estimate information in a coding-agent tool observation.",
            f"Tool type: {request.tool_type}",
        ]
        if query:
            parts.append(f"Current goal and context: {query}")
        parts.append("Candidate observation:")
        return "\n".join(parts) + "\n"

    def _anchors(self, request: PruningRequest) -> set[int]:
        anchors: set[int] = set()
        if self.config.protect_structure:
            anchors.update(structural_anchor_lines(request.lines))
        if self.config.protect_errors:
            anchors.update(error_anchor_lines(request.lines))
        if self.config.protect_metadata_anchors:
            anchors.update(_metadata_anchor_lines(request.metadata))
        return {line_no for line_no in anchors if 1 <= line_no <= len(request.lines)}

    def prune(self, request: PruningRequest) -> PruningResult:
        started = time.perf_counter()
        lines = request.lines
        line_count = len(lines)
        target = request.budget.target_lines(line_count)
        if target >= line_count:
            kept = tuple(range(1, line_count + 1))
            return PruningResult(
                method=self.name,
                original_line_count=line_count,
                kept_line_numbers=kept,
                pruned_text=render_pruned_text(
                    lines,
                    kept,
                    show_line_numbers=self.config.show_line_numbers,
                ),
                line_scores=tuple(
                    LineScore(line_no=index, score=0.0, reasons=("no-prune",)) for index in kept
                ),
                latency_ms=(time.perf_counter() - started) * 1000.0,
                metadata={
                    "target_lines": target,
                    "scorer_calls": 0,
                    "model_forward_count": 0,
                },
                request_id=request.request_id,
            )
        if self.scorer is None:
            raise RuntimeError(
                "conditional_ppl needs a scorer; set model_path in config "
                "or inject a ConditionalSurprisalScorer"
            )

        blocks = _candidate_blocks(request.text, self.config.block_max_lines)
        base_context = self._base_context(request)
        block_scores: list[float] = []
        scorer_calls = 0
        for block in blocks:
            score = self.scorer.score(
                base_context,
                block.text or "\n",
                first_token_only=self.config.coarse_first_token_only,
            )
            block_scores.append(self._checked_score(score))
            scorer_calls += 1

        if self.config.coarse_top_blocks is not None:
            fine_block_count = min(
                len(blocks),
                self.config.coarse_top_blocks,
            )
        else:
            fine_block_count = min(
                len(blocks),
                max(1, math.ceil(len(blocks) * self.config.coarse_top_fraction)),
            )
        ranked_blocks = sorted(
            range(len(blocks)),
            key=lambda index: (
                -block_scores[index],
                blocks[index].start_line,
                blocks[index].end_line,
            ),
        )
        fine_block_indexes = set(ranked_blocks[:fine_block_count])

        anchors = self._anchors(request)
        for index, block in enumerate(blocks):
            if anchors.intersection(block.line_numbers):
                fine_block_indexes.add(index)

        coarse_by_line = [0.0] * line_count
        coarse_reasons: list[list[str]] = [[] for _ in lines]
        for index, block in enumerate(blocks):
            for line_no in block.line_numbers:
                if not 1 <= line_no <= line_count:
                    continue
                if (
                    not coarse_reasons[line_no - 1]
                    or block_scores[index] > coarse_by_line[line_no - 1]
                ):
                    coarse_by_line[line_no - 1] = block_scores[index]
                coarse_reasons[line_no - 1].append(f"coarse:{block.start_line}-{block.end_line}")

        fine_lines = {
            line_no
            for index in fine_block_indexes
            for line_no in blocks[index].line_numbers
            if 1 <= line_no <= line_count
        }
        fine_scores: dict[int, float] = {}
        for line_no in sorted(fine_lines):
            context_start = max(1, line_no - self.config.fine_context_lines)
            preceding = "\n".join(lines[context_start - 1 : line_no - 1])
            context = base_context
            if preceding:
                context += f"Immediately preceding lines:\n{preceding}\n"
            continuation = lines[line_no - 1] or "\n"
            score = self.scorer.score(
                context,
                continuation,
                first_token_only=self.config.first_token_only,
            )
            fine_scores[line_no] = self._checked_score(score)
            scorer_calls += 1

        weight_sum = self.config.coarse_weight + self.config.fine_weight
        scores: list[float] = []
        reasons: list[tuple[str, ...]] = []
        for line_no in range(1, line_count + 1):
            fine = fine_scores.get(line_no)
            if fine is None:
                score = self.config.coarse_weight * coarse_by_line[line_no - 1]
                reason = list(coarse_reasons[line_no - 1])
                reason.append("coarse-only")
            else:
                score = (
                    self.config.coarse_weight * coarse_by_line[line_no - 1]
                    + self.config.fine_weight * fine
                )
                reason = list(coarse_reasons[line_no - 1])
                reason.append("fine-surprisal")
            score /= weight_sum
            if line_no in anchors:
                reason.append("protected-anchor")
            scores.append(score)
            reasons.append(tuple(reason))

        kept = select_line_numbers(
            scores,
            request.budget,
            mandatory=anchors,
            expansion_seeds=anchors if self.config.expand_anchor_context else (),
        )
        return PruningResult(
            method=self.name,
            original_line_count=line_count,
            kept_line_numbers=kept,
            pruned_text=render_pruned_text(
                lines,
                kept,
                show_line_numbers=self.config.show_line_numbers,
            ),
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
                "target_lines": target,
                "block_count": len(blocks),
                "fine_block_count": len(fine_block_indexes),
                "fine_line_count": len(fine_lines),
                "scorer_calls": scorer_calls,
                "model_forward_count": scorer_calls,
                "scorer": getattr(self.scorer, "name", type(self.scorer).__name__),
                "coarse_first_token_only": self.config.coarse_first_token_only,
                "fine_first_token_only": self.config.first_token_only,
                "anchor_lines": sorted(anchors),
            },
            request_id=request.request_id,
        )


__all__ = [
    "ConditionalPPLConfig",
    "ConditionalPPLPruner",
    "ConditionalSurprisalScorer",
    "HFConditionalSurprisalScorer",
]
