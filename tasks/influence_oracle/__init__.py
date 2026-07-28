"""Likelihood-preserving leave-one-out and greedy pruning oracle."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .pruner import (
    HFLogLikelihoodScorer,
    InfluenceOracleConfig,
    InfluenceOraclePruner,
    NextActionObjective,
)


def build_pruner(
    config: Mapping[str, Any] | None = None,
) -> InfluenceOraclePruner:
    """Build an offline oracle from a JSON-compatible configuration."""

    payload = dict(config or {})
    scorer_payload = dict(payload.pop("scorer", {}))
    objective_payload = dict(payload.pop("objective", {}))
    pruner_payload = dict(payload.pop("pruner", {}))

    model_path = payload.pop("model_path", None)
    if model_path is None:
        model_path = scorer_payload.pop("model_path", None)
    if payload:
        unknown = ", ".join(sorted(payload))
        raise ValueError(f"unknown influence_oracle config keys: {unknown}")

    scorer = (
        None
        if model_path is None
        else HFLogLikelihoodScorer(
            model_path=str(model_path),
            **scorer_payload,
        )
    )
    if model_path is None and scorer_payload:
        raise ValueError("scorer options require model_path")
    objective = NextActionObjective(**objective_payload)
    return InfluenceOraclePruner(
        scorer=scorer,
        objective=objective,
        config=InfluenceOracleConfig(**pruner_payload),
    )


__all__ = [
    "HFLogLikelihoodScorer",
    "InfluenceOracleConfig",
    "InfluenceOraclePruner",
    "NextActionObjective",
    "build_pruner",
]
