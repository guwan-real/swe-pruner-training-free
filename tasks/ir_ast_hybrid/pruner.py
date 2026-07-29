from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from tasks.execution_ast.pruner import (
    ExecutionASTConfig,
    ExecutionASTPruner,
)
from tasks.ir_structural.pruner import (
    IRStructuralConfig,
    IRStructuralPruner,
)
from tf_pruning.protocol import (
    PruningRequest,
    PruningResult,
    coerce_line_scores,
)
from tf_pruning.selection import render_pruned_text, select_line_numbers


def _rank_percentiles(scores: Sequence[float]) -> list[float]:
    """Map arbitrary component scores to deterministic [0, 1] rank percentiles."""

    if not scores:
        return []
    unique_scores = sorted({float(score) for score in scores}, reverse=True)
    if len(unique_scores) == 1:
        return [1.0]
    denominator = len(unique_scores) - 1
    percentile_by_score = {
        score: 1.0 - rank / denominator for rank, score in enumerate(unique_scores)
    }
    return [percentile_by_score[float(score)] for score in scores]


@dataclass(frozen=True)
class IRASTHybridConfig:
    """Static rank-fusion settings; no weights are fitted."""

    ir_weight: float = 0.55
    execution_ast_weight: float = 0.45
    mandatory_reason_prefixes: tuple[str, ...] = (
        "query_symbol_match",
        "source_error",
        "traceback_",
        "test_",
        "diff_",
        "grep_hit",
        "tree_query_hit",
    )
    show_line_numbers: bool = False
    ir: IRStructuralConfig = IRStructuralConfig(show_line_numbers=False)
    execution_ast: ExecutionASTConfig = ExecutionASTConfig(show_line_numbers=False)

    def __post_init__(self) -> None:
        if self.ir_weight < 0 or self.execution_ast_weight < 0:
            raise ValueError("hybrid weights must be non-negative")
        if self.ir_weight + self.execution_ast_weight <= 0:
            raise ValueError("at least one hybrid weight must be positive")

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, Any] | None,
    ) -> "IRASTHybridConfig":
        if config is None:
            return cls()
        values = dict(config)
        weights = values.pop("weights", None)
        if weights is not None:
            if not isinstance(weights, Mapping):
                raise TypeError("weights must be a mapping")
            values.setdefault("ir_weight", weights.get("ir", cls.ir_weight))
            values.setdefault(
                "execution_ast_weight",
                weights.get("execution_ast", cls.execution_ast_weight),
            )
        ir_config = values.pop("ir", None)
        ast_config = values.pop("execution_ast", values.pop("ast", None))
        if "mandatory_reason_prefixes" in values:
            values["mandatory_reason_prefixes"] = tuple(
                str(item) for item in values["mandatory_reason_prefixes"]
            )
        known = {
            "ir_weight",
            "execution_ast_weight",
            "mandatory_reason_prefixes",
            "show_line_numbers",
        }
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"unknown IR+AST hybrid config keys: {unknown}")
        return cls(
            **values,
            ir=IRStructuralConfig.from_mapping(ir_config),
            execution_ast=ExecutionASTConfig.from_mapping(ast_config),
        )


class IRASTHybridPruner:
    """Fuse two training-free rankers while protecting execution evidence."""

    name = "ir_ast_hybrid"

    def __init__(self, config: IRASTHybridConfig | None = None) -> None:
        self.config = config or IRASTHybridConfig()
        self.ir_pruner = IRStructuralPruner(self.config.ir)
        self.ast_pruner = ExecutionASTPruner(self.config.execution_ast)

    def prune(self, request: PruningRequest) -> PruningResult:
        started_at = time.perf_counter()
        if not request.lines:
            return PruningResult(
                method=self.name,
                original_line_count=0,
                kept_line_numbers=(),
                pruned_text="",
                latency_ms=(time.perf_counter() - started_at) * 1000.0,
                metadata={"config": asdict(self.config), "training_free": True},
                request_id=request.request_id,
            )

        ir_result = self.ir_pruner.prune(request)
        ast_result = self.ast_pruner.prune(request)
        ir_raw = [item.score for item in ir_result.line_scores]
        ast_raw = [item.score for item in ast_result.line_scores]
        ir_ranks = _rank_percentiles(ir_raw)
        ast_ranks = _rank_percentiles(ast_raw)
        weight_sum = self.config.ir_weight + self.config.execution_ast_weight
        fused = [
            (self.config.ir_weight * ir_score + self.config.execution_ast_weight * ast_score)
            / weight_sum
            for ir_score, ast_score in zip(ir_ranks, ast_ranks)
        ]

        reasons: dict[int, list[str]] = {}
        mandatory: set[int] = set()
        expansion_seeds: set[int] = set()
        for line_no, (ir_score, ast_score) in enumerate(
            zip(ir_result.line_scores, ast_result.line_scores),
            start=1,
        ):
            line_reasons: list[str] = []
            if ir_score.score > 0:
                line_reasons.append("ir_rank")
            if ast_score.score > self.config.execution_ast.fallback_score:
                line_reasons.append("execution_ast_rank")
            line_reasons.extend(f"ir:{reason}" for reason in ir_score.reasons)
            line_reasons.extend(f"ast:{reason}" for reason in ast_score.reasons)
            reasons[line_no] = line_reasons
            if any(
                reason.startswith(prefix)
                for reason in ast_score.reasons
                for prefix in self.config.mandatory_reason_prefixes
            ):
                mandatory.add(line_no)
                expansion_seeds.add(line_no)

        kept = select_line_numbers(
            fused,
            request.budget,
            mandatory=mandatory,
            expansion_seeds=expansion_seeds,
        )
        return PruningResult(
            method=self.name,
            original_line_count=len(request.lines),
            kept_line_numbers=kept,
            pruned_text=render_pruned_text(
                request.lines,
                kept,
                show_line_numbers=self.config.show_line_numbers,
            ),
            line_scores=coerce_line_scores(fused, reasons),
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
            metadata={
                "config": asdict(self.config),
                "component_methods": [ir_result.method, ast_result.method],
                "component_latency_ms": {
                    ir_result.method: ir_result.latency_ms,
                    ast_result.method: ast_result.latency_ms,
                },
                "mandatory_lines": sorted(mandatory),
                "detected_tool_type": ast_result.metadata.get("detected_tool_type"),
                "training_free": True,
                "model_forward_count": 0,
            },
            request_id=request.request_id,
        )
