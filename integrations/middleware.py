from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from tf_pruning.budgets import DEFAULT_LENGTH_AWARE_BUDGET, LengthAwareBudget
from tf_pruning.protocol import Pruner, PruningRequest, PruningResult


@dataclass(frozen=True)
class MiddlewareOutcome:
    text: str
    pruned: bool
    result: PruningResult | None
    error: str | None = None


def infer_tool_type(
    *,
    tool_name: str | None = None,
    command: str | None = None,
    path: str | None = None,
) -> str:
    combined = " ".join(item.lower() for item in (tool_name, command) if item)
    if any(token in combined for token in ("traceback", "stack trace")):
        return "traceback"
    if any(token in combined for token in ("git diff", "show diff", "patch")):
        return "diff"
    if any(token in combined for token in ("pytest", "unittest", "npm test", "cargo test", "test")):
        return "test_log"
    if any(token in combined for token in ("grep", "ripgrep", "rg ", "search", "find symbol")):
        return "grep"
    if any(token in combined for token in ("tree", "list directory", "ls ")):
        return "tree"
    if any(token in combined for token in ("cat", "read", "open file", "view file")):
        return "source"
    if path:
        return "source"
    return "auto"


class TrainingFreeMiddleware:
    """Fail-open adapter called after a coding-agent tool returns text."""

    def __init__(
        self,
        pruner: Pruner,
        *,
        budget_schedule: LengthAwareBudget | None = None,
        fail_open: bool = True,
    ) -> None:
        self.pruner = pruner
        self.budget_schedule = budget_schedule or DEFAULT_LENGTH_AWARE_BUDGET
        self.fail_open = fail_open

    def prune_tool_response(
        self,
        text: str,
        *,
        query: str = "",
        tool_type: str = "auto",
        tool_name: str | None = None,
        command: str | None = None,
        path: str | None = None,
        recent_context: tuple[str, ...] = (),
        metadata: Mapping[str, Any] | None = None,
        request_id: str | None = None,
    ) -> MiddlewareOutcome:
        lines = text.splitlines()
        resolved_tool_type = (
            infer_tool_type(
                tool_name=tool_name,
                command=command,
                path=path,
            )
            if tool_type == "auto"
            else tool_type
        )
        request = PruningRequest(
            text=text,
            query=query,
            tool_type=resolved_tool_type,
            path=path,
            recent_context=recent_context,
            budget=self.budget_schedule.for_line_count(len(lines)),
            metadata=dict(metadata or {}),
            request_id=request_id,
        )
        try:
            result = self.pruner.prune(request)
        except Exception as exc:
            if not self.fail_open:
                raise
            return MiddlewareOutcome(
                text=text,
                pruned=False,
                result=None,
                error=f"{type(exc).__name__}: {exc}",
            )
        changed = result.kept_line_count < result.original_line_count
        return MiddlewareOutcome(
            text=result.pruned_text if changed else text,
            pruned=changed,
            result=result,
        )
