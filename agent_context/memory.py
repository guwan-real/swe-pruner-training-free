from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from agent_context.engine import ContextEngine


@dataclass(frozen=True)
class MemoryToolRequest:
    operation: str
    observation_id: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class MemoryToolResponse:
    operation: str
    observation_id: str
    content: str
    metadata: Mapping[str, Any]


class ObservationMemoryTools:
    """Agent-neutral retrieval API; adapters decide how tools are exposed."""

    def __init__(self, engine: ContextEngine) -> None:
        self.engine = engine

    def execute(self, request: MemoryToolRequest) -> MemoryToolResponse:
        operation = request.operation.strip().lower()
        arguments = dict(request.arguments)
        if operation == "read":
            content = self.engine.read(
                request.observation_id,
                start_line=int(arguments.get("start_line", 1)),
                end_line=(
                    None if arguments.get("end_line") is None else int(arguments["end_line"])
                ),
            )
            return MemoryToolResponse(
                operation=operation,
                observation_id=request.observation_id,
                content=content,
                metadata={
                    "delivery": "append-only",
                    "historical_slot_rewritten": False,
                },
            )
        if operation == "search":
            pattern = str(arguments.get("pattern", ""))
            matches = self.engine.search(
                request.observation_id,
                pattern,
                regex=bool(arguments.get("regex", False)),
                max_results=int(arguments.get("max_results", 50)),
            )
            content = "\n".join(f"{line_no}:{line}" for line_no, line in matches)
            return MemoryToolResponse(
                operation=operation,
                observation_id=request.observation_id,
                content=content,
                metadata={
                    "matches": len(matches),
                    "delivery": "append-only",
                    "historical_slot_rewritten": False,
                },
            )
        if operation == "pin":
            self.engine.pin(request.observation_id)
            return MemoryToolResponse(
                operation=operation,
                observation_id=request.observation_id,
                content=f"Pinned {request.observation_id}",
                metadata={"pinned": True},
            )
        if operation == "unpin":
            self.engine.unpin(request.observation_id)
            return MemoryToolResponse(
                operation=operation,
                observation_id=request.observation_id,
                content=f"Unpinned {request.observation_id}",
                metadata={"pinned": False},
            )
        raise ValueError(f"unknown memory operation: {request.operation}")
