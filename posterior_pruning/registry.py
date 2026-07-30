from __future__ import annotations

from typing import Any, Mapping

from posterior_pruning.candidates import CandidateConfig
from posterior_pruning.methods.block_influence import (
    BlockInfluenceConfig,
    BlockInfluenceMethod,
)
from posterior_pruning.methods.budget_search import BudgetSearchConfig, BudgetSearchMethod
from posterior_pruning.methods.common import AcceptanceConfig
from posterior_pruning.methods.greedy_blocks import GreedyBlocksConfig, GreedyBlocksMethod
from posterior_pruning.methods.single_verify import SingleVerifyConfig, SingleVerifyMethod
from posterior_pruning.scoring import ActionScorer

METHODS = ("single_verify", "budget_search", "greedy_blocks", "block_influence")


def _float_tuple(value: Any, *, name: str) -> tuple[float, ...]:
    if isinstance(value, str):
        raw = value.replace(",", " ").split()
    elif isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        raise ValueError(f"{name} must be a comma-separated string or array")
    result = tuple(float(item) for item in raw)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def build_method(
    name: str,
    scorer: ActionScorer,
    config: Mapping[str, Any] | None = None,
) -> Any:
    values = dict(config or {})
    acceptance = AcceptanceConfig(
        max_mean_logprob_drop=float(values.get("max_mean_logprob_drop", 0.08))
    )
    candidates = CandidateConfig(
        block_max_lines=int(values.get("block_max_lines", 12)),
        protect_errors=bool(values.get("protect_errors", True)),
        protect_diffs=bool(values.get("protect_diffs", True)),
        protect_edge_lines=bool(values.get("protect_edge_lines", True)),
    )
    if name == "single_verify":
        return SingleVerifyMethod(
            scorer,
            SingleVerifyConfig(acceptance=acceptance, candidates=candidates),
        )
    if name == "budget_search":
        return BudgetSearchMethod(
            scorer,
            BudgetSearchConfig(
                ratios=_float_tuple(
                    values.get("ratios", (0.25, 0.4, 0.6, 0.8)),
                    name="ratios",
                ),
                max_candidates=int(values.get("max_candidates", 4)),
                acceptance=acceptance,
                candidates=candidates,
            ),
        )
    if name == "greedy_blocks":
        return GreedyBlocksMethod(
            scorer,
            GreedyBlocksConfig(
                max_evaluations=int(values.get("max_evaluations", 6)),
                acceptance=acceptance,
                candidates=candidates,
            ),
        )
    if name == "block_influence":
        return BlockInfluenceMethod(
            scorer,
            BlockInfluenceConfig(
                max_block_evaluations=int(values.get("max_block_evaluations", 6)),
                acceptance=acceptance,
                candidates=candidates,
            ),
        )
    raise ValueError(f"unknown posterior method {name!r}; choose from {', '.join(METHODS)}")
