"""Rank-fusion hybrid of IR relevance and execution/AST evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .pruner import IRASTHybridConfig, IRASTHybridPruner


def build_pruner(
    config: Mapping[str, Any] | None = None,
) -> IRASTHybridPruner:
    return IRASTHybridPruner(IRASTHybridConfig.from_mapping(config))


__all__ = [
    "IRASTHybridConfig",
    "IRASTHybridPruner",
    "build_pruner",
]
