"""Training-free line pruning from frozen-backbone hidden states."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .pruner import HiddenStateSimilarityConfig, HiddenStateSimilarityPruner

__all__ = [
    "HiddenStateSimilarityConfig",
    "HiddenStateSimilarityPruner",
    "build_pruner",
]


def build_pruner(
    config: Mapping[str, Any] | None = None,
) -> HiddenStateSimilarityPruner:
    """Build the task pruner for a unified experiment runner."""

    return HiddenStateSimilarityPruner(config)
