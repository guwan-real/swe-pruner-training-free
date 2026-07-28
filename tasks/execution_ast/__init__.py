"""Execution-signal rules with Python AST skeleton preservation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .pruner import (
    ExecutionASTConfig,
    ExecutionASTPruner,
    detect_tool_type,
)


def build_pruner(
    config: Mapping[str, Any] | None = None,
) -> ExecutionASTPruner:
    """Build the task through the common experiment-factory convention."""

    return ExecutionASTPruner(ExecutionASTConfig.from_mapping(config))


__all__ = [
    "ExecutionASTConfig",
    "ExecutionASTPruner",
    "build_pruner",
    "detect_tool_type",
]
