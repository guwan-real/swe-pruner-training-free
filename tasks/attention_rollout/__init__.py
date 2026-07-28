"""Training-free pruning from attention mass or attention rollout."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .pruner import AttentionPrunerConfig, AttentionRolloutPruner

__all__ = ["AttentionPrunerConfig", "AttentionRolloutPruner", "build_pruner"]


def build_pruner(
    config: Mapping[str, Any] | None = None,
) -> AttentionRolloutPruner:
    """Build the task pruner for a unified experiment runner."""

    return AttentionRolloutPruner(config)
