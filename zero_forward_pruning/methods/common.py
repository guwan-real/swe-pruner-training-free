from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Mapping, Sequence
from urllib.parse import quote

from zero_forward_pruning.blocks import (
    EvidenceBlock,
    build_blocks,
    expand_indices,
    selected_lines,
)
from zero_forward_pruning.protocol import PruningRequest, PruningResult
from zero_forward_pruning.ranking import discriminative_intent_terms
from zero_forward_pruning.store import RawStore
from zero_forward_pruning.text import (
    OutputKind,
    classify_output,
    estimate_tokens,
    identifiers,
    terms,
)


@dataclass(frozen=True)
class PrunerConfig:
    min_input_tokens: int = 1500
    min_savings_tokens: int = 256
    max_retention_ratio: float = 0.85
    max_cpu_ms: float = 50.0
    max_output_chars: int = 9000
    block_max_lines: int = 16
    public_base_url: str = "http://host.docker.internal:8121"
    raw_store: RawStore | None = None
    require_recovery: bool = True

    def __post_init__(self) -> None:
        if self.min_input_tokens < 0:
            raise ValueError("min_input_tokens must be non-negative")
        if self.min_savings_tokens < 1:
            raise ValueError("min_savings_tokens must be positive")
        if not 0.0 < self.max_retention_ratio < 1.0:
            raise ValueError("max_retention_ratio must be in (0, 1)")
        if self.max_cpu_ms <= 0:
            raise ValueError("max_cpu_ms must be positive")
        if self.max_output_chars < 1000:
            raise ValueError("max_output_chars must be at least 1000")
        if self.block_max_lines < 1:
            raise ValueError("block_max_lines must be positive")
        if self.require_recovery and self.raw_store is None:
            raise ValueError("require_recovery needs a raw_store")


@dataclass(frozen=True)
class Selection:
    block_indices: frozenset[int]
    block_scores: Mapping[int, float] = field(default_factory=dict)
    diagnostics: Mapping[str, object] = field(default_factory=dict)


def unchanged_result(
    method: str,
    request: PruningRequest,
    *,
    status: str,
    started_at: float,
    reason: str,
    kind: OutputKind | None = None,
    error: str | None = None,
    diagnostics: Mapping[str, object] | None = None,
) -> PruningResult:
    tokens = estimate_tokens(request.code)
    line_count = len(request.code.splitlines())
    details: dict[str, object] = {"reason": reason}
    if kind is not None:
        details["output_kind"] = kind.value
    details.update(diagnostics or {})
    return PruningResult(
        pruned_code=request.code,
        origin_token_cnt=tokens,
        left_token_cnt=tokens,
        model_input_token_cnt=0,
        method=method,
        status=status,
        original_line_count=line_count,
        kept_line_count=line_count,
        retention_ratio=1.0,
        latency_ms=(time.perf_counter() - started_at) * 1000,
        kept_line_numbers=tuple(range(1, line_count + 1)),
        error=error,
        diagnostics=details,
    )


def _render_compacted(
    lines: Sequence[str],
    kept_lines: set[int],
    *,
    raw_id: str,
    recovery_url: str,
) -> str:
    recovery_path = f"/tmp/zero-forward-{raw_id}.txt"
    header = (
        f'<zero_forward_compaction kept_lines="{len(kept_lines)}" '
        f'original_lines="{len(lines)}" raw_id="{raw_id}">\n'
        f"Recover the exact full output to a file with: "
        f"curl -fsS '{recovery_url}' -o '{recovery_path}'\n"
        "</zero_forward_compaction>"
    )
    output = [header]
    line_no = 1
    while line_no <= len(lines):
        if line_no in kept_lines:
            output.append(lines[line_no - 1])
            line_no += 1
            continue
        start = line_no
        while line_no <= len(lines) and line_no not in kept_lines:
            line_no += 1
        end = line_no - 1
        label = f"line {start}" if start == end else f"lines {start}-{end}"
        output.append(f"... [zero-forward omitted {label}; use recovery command above] ...")
    return "\n".join(output)


def _line_scores(
    blocks: Sequence[EvidenceBlock],
    block_scores: Mapping[int, float],
    line_count: int,
) -> tuple[float, ...]:
    values = [0.0] * line_count
    maximum = max(block_scores.values(), default=0.0)
    for block in blocks:
        score = block_scores.get(block.index, 0.0)
        normalized = score / maximum if maximum > 0 else 0.0
        if block.reasons:
            normalized = max(normalized, 1.0)
        for line_no in block.line_numbers:
            values[line_no - 1] = min(1.0, normalized)
    return tuple(values)


def _cap_block_selection(
    *,
    request: PruningRequest,
    blocks: Sequence[EvidenceBlock],
    selected: set[int],
    lines: Sequence[str],
    raw_id: str,
    recovery_url: str,
    max_output_chars: int,
    block_scores: Mapping[int, float],
    use_intent: bool,
) -> tuple[set[int], dict[str, object]]:
    """Fit selected evidence below mini-swe-agent's long-output display limit."""

    query_terms = set(terms(request.intent_text)) if use_intent else set()
    discriminative = (
        discriminative_intent_terms(blocks, request.intent_text) if use_intent else set()
    )
    query_identifiers = identifiers(request.intent_text) if use_intent else set()
    directly_relevant = {
        block.index for block in blocks if discriminative.intersection(block.terms)
    }
    relevant_neighbourhood = expand_indices(
        directly_relevant,
        block_count=len(blocks),
        radius=2,
    )
    critical_reasons = {
        "error",
        "trace-frame",
        "source-location",
        "test-failure",
        "diff",
    }

    def priority(block: EvidenceBlock) -> tuple[float, ...]:
        return (
            float(len(critical_reasons.intersection(block.reasons))),
            float(len(query_identifiers.intersection(block.identifiers))),
            float(len(query_terms.intersection(block.terms))),
            float(block.index in relevant_neighbourhood),
            float(block.index in {0, len(blocks) - 1}),
            float("structure" in block.reasons),
            float(block_scores.get(block.index, 0.0)),
            float(-block.index),
        )

    ordered = sorted(
        (block for block in blocks if block.index in selected),
        key=priority,
        reverse=True,
    )
    empty_render = _render_compacted(
        lines,
        set(),
        raw_id=raw_id,
        recovery_url=recovery_url,
    )
    remaining = max(0, max_output_chars - len(empty_render) - 128)
    capped: set[int] = set()
    included: list[EvidenceBlock] = []
    for block in ordered:
        # A conservative marker allowance accounts for the omission range that
        # may be split on either side of this block.
        estimated_cost = len(block.text) + 1 + 96
        if estimated_cost > remaining:
            continue
        capped.add(block.index)
        included.append(block)
        remaining -= estimated_cost
    compacted = _render_compacted(
        lines,
        set(selected_lines(blocks, capped)),
        raw_id=raw_id,
        recovery_url=recovery_url,
    )
    while len(compacted) > max_output_chars and included:
        removed = included.pop()
        capped.remove(removed.index)
        compacted = _render_compacted(
            lines,
            set(selected_lines(blocks, capped)),
            raw_id=raw_id,
            recovery_url=recovery_url,
        )
    return capped, {
        "output_char_cap_applied": True,
        "max_output_chars": max_output_chars,
        "pre_cap_block_count": len(selected),
        "post_cap_block_count": len(capped),
    }


class BaseZeroForwardPruner:
    name = "base"

    def __init__(self, config: PrunerConfig):
        self.config = config

    def select(
        self,
        request: PruningRequest,
        blocks: Sequence[EvidenceBlock],
        kind: OutputKind,
    ) -> Selection:
        raise NotImplementedError

    def prune(self, request: PruningRequest) -> PruningResult:
        started_at = time.perf_counter()
        kind = classify_output(request.code, command=request.command, path=request.path)
        original_tokens = estimate_tokens(request.code)
        if not request.code.strip():
            return unchanged_result(
                self.name,
                request,
                status="skipped",
                started_at=started_at,
                reason="empty-output",
                kind=kind,
            )
        if original_tokens < self.config.min_input_tokens:
            return unchanged_result(
                self.name,
                request,
                status="skipped",
                started_at=started_at,
                reason="below-min-input-tokens",
                kind=kind,
                diagnostics={"min_input_tokens": self.config.min_input_tokens},
            )
        if kind == OutputKind.DIFF:
            return unchanged_result(
                self.name,
                request,
                status="skipped",
                started_at=started_at,
                reason="diff-is-never-pruned",
                kind=kind,
            )
        stored_id: str | None = None
        try:
            blocks = build_blocks(
                request.code,
                kind=kind,
                max_lines=self.config.block_max_lines,
            )
            selection = self.select(request, blocks, kind)
            kept = set(selected_lines(blocks, selection.block_indices))
            lines = request.code.splitlines()
            if not kept or len(kept) >= len(lines):
                return unchanged_result(
                    self.name,
                    request,
                    status="skipped",
                    started_at=started_at,
                    reason="no-safe-reduction",
                    kind=kind,
                    diagnostics=selection.diagnostics,
                )
            line_retention = len(kept) / len(lines)
            if line_retention > self.config.max_retention_ratio:
                return unchanged_result(
                    self.name,
                    request,
                    status="skipped",
                    started_at=started_at,
                    reason="retention-above-cost-gate",
                    kind=kind,
                    diagnostics={
                        **selection.diagnostics,
                        "line_retention": line_retention,
                        "max_retention_ratio": self.config.max_retention_ratio,
                    },
                )
            if self.config.raw_store is None:
                return unchanged_result(
                    self.name,
                    request,
                    status="skipped",
                    started_at=started_at,
                    reason="recovery-store-unavailable",
                    kind=kind,
                    diagnostics=selection.diagnostics,
                )
            stored = self.config.raw_store.save(
                request.code,
                {
                    "method": self.name,
                    "request_id": request.request_id,
                    "output_kind": kind.value,
                    "command": request.command[:2000],
                    "path": request.path,
                },
            )
            stored_id = stored.raw_id
            base_url = self.config.public_base_url.rstrip("/")
            recovery_url = f"{base_url}/raw/{quote(stored.raw_id, safe='')}"
            compacted = _render_compacted(
                lines,
                kept,
                raw_id=stored.raw_id,
                recovery_url=recovery_url,
            )
            cap_diagnostics: dict[str, object] = {
                "output_char_cap_applied": False,
                "max_output_chars": self.config.max_output_chars,
            }
            if len(compacted) > self.config.max_output_chars:
                capped_indices, cap_diagnostics = _cap_block_selection(
                    request=request,
                    blocks=blocks,
                    selected=set(selection.block_indices),
                    lines=lines,
                    raw_id=stored.raw_id,
                    recovery_url=recovery_url,
                    max_output_chars=self.config.max_output_chars,
                    block_scores=selection.block_scores,
                    use_intent=self.name != "safe_rules",
                )
                kept = set(selected_lines(blocks, capped_indices))
                if not kept:
                    self.config.raw_store.delete(stored.raw_id)
                    return unchanged_result(
                        self.name,
                        request,
                        status="skipped",
                        started_at=started_at,
                        reason="output-cap-left-no-evidence",
                        kind=kind,
                        diagnostics={
                            **selection.diagnostics,
                            **cap_diagnostics,
                        },
                    )
                compacted = _render_compacted(
                    lines,
                    kept,
                    raw_id=stored.raw_id,
                    recovery_url=recovery_url,
                )
            left_tokens = estimate_tokens(compacted)
            savings = original_tokens - left_tokens
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            if savings < self.config.min_savings_tokens:
                self.config.raw_store.delete(stored.raw_id)
                return unchanged_result(
                    self.name,
                    request,
                    status="skipped",
                    started_at=started_at,
                    reason="insufficient-token-savings",
                    kind=kind,
                    diagnostics={
                        **selection.diagnostics,
                        **cap_diagnostics,
                        "estimated_savings_tokens": savings,
                        "min_savings_tokens": self.config.min_savings_tokens,
                    },
                )
            if elapsed_ms > self.config.max_cpu_ms:
                self.config.raw_store.delete(stored.raw_id)
                return unchanged_result(
                    self.name,
                    request,
                    status="skipped",
                    started_at=started_at,
                    reason="cpu-budget-exceeded",
                    kind=kind,
                    diagnostics={
                        **selection.diagnostics,
                        **cap_diagnostics,
                        "measured_cpu_ms": elapsed_ms,
                        "max_cpu_ms": self.config.max_cpu_ms,
                    },
                )
            diagnostics = {
                **selection.diagnostics,
                **cap_diagnostics,
                "output_kind": kind.value,
                "estimated_savings_tokens": savings,
                "cost_gate_passed": True,
                "reversible": True,
                "threshold_used_for_model_scoring": False,
            }
            return PruningResult(
                pruned_code=compacted,
                origin_token_cnt=original_tokens,
                left_token_cnt=left_tokens,
                model_input_token_cnt=0,
                method=self.name,
                status="pruned",
                original_line_count=len(lines),
                kept_line_count=len(kept),
                retention_ratio=left_tokens / original_tokens,
                latency_ms=elapsed_ms,
                raw_id=stored.raw_id,
                recovery_url=recovery_url,
                kept_line_numbers=tuple(sorted(kept)),
                score=_line_scores(blocks, selection.block_scores, len(lines)),
                diagnostics=diagnostics,
            )
        except Exception as exc:
            if stored_id and self.config.raw_store is not None:
                self.config.raw_store.delete(stored_id)
            return unchanged_result(
                self.name,
                request,
                status="error",
                started_at=started_at,
                reason="fail-open",
                kind=kind,
                error=str(exc),
            )


def ratio_target_lines(request: PruningRequest, line_count: int) -> int:
    """Use the legacy threshold only as an ablation budget, never for model scoring."""

    keep_ratio = min(0.9, max(0.1, 1.0 - request.threshold))
    return max(1, math.ceil(line_count * keep_ratio))
