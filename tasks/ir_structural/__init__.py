"""Training-free sparse retrieval with structural anchors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .pruner import IRStructuralConfig, IRStructuralPruner


def build_pruner(
    config: Mapping[str, Any] | None = None,
) -> IRStructuralPruner:
    """Build the task through the common experiment-factory convention."""

    return IRStructuralPruner(IRStructuralConfig.from_mapping(config))


__all__ = [
    "IRStructuralConfig",
    "IRStructuralPruner",
    "build_pruner",
]
