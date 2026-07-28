"""Coarse-to-fine conditional-surprisal pruning."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .pruner import (
    ConditionalPPLConfig,
    ConditionalPPLPruner,
    HFConditionalSurprisalScorer,
)


def build_pruner(
    config: Mapping[str, Any] | None = None,
) -> ConditionalPPLPruner:
    """Build the task pruner from a JSON-compatible configuration.

    ``model_path`` may be supplied at the top level or under ``scorer``.
    Constructing the pruner never imports Transformers or loads model weights.
    """

    payload = dict(config or {})
    scorer_payload = dict(payload.pop("scorer", {}))
    pruner_payload = dict(payload.pop("pruner", {}))

    model_path = payload.pop("model_path", None)
    if model_path is None:
        model_path = scorer_payload.pop("model_path", None)
    if payload:
        unknown = ", ".join(sorted(payload))
        raise ValueError(f"unknown conditional_ppl config keys: {unknown}")

    scorer = (
        None
        if model_path is None
        else HFConditionalSurprisalScorer(
            model_path=str(model_path),
            **scorer_payload,
        )
    )
    if model_path is None and scorer_payload:
        raise ValueError("scorer options require model_path")
    return ConditionalPPLPruner(
        scorer=scorer,
        config=ConditionalPPLConfig(**pruner_payload),
    )


__all__ = [
    "ConditionalPPLConfig",
    "ConditionalPPLPruner",
    "HFConditionalSurprisalScorer",
    "build_pruner",
]
