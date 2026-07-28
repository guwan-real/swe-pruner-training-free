from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class BudgetConfig:
    """Line-level output budget shared by every experiment."""

    keep_ratio: float = 0.5
    min_lines: int = 1
    max_lines: int | None = None
    no_prune_below: int = 20
    context_window: int = 1

    def __post_init__(self) -> None:
        if not 0.0 < self.keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1]")
        if self.min_lines < 0:
            raise ValueError("min_lines must be non-negative")
        if self.max_lines is not None and self.max_lines < 0:
            raise ValueError("max_lines must be non-negative")
        if self.no_prune_below < 0:
            raise ValueError("no_prune_below must be non-negative")
        if self.context_window < 0:
            raise ValueError("context_window must be non-negative")

    def target_lines(self, line_count: int) -> int:
        if line_count <= self.no_prune_below:
            return line_count
        target = max(self.min_lines, round(line_count * self.keep_ratio))
        if self.max_lines is not None:
            target = min(target, self.max_lines)
        return max(0, min(line_count, target))


@dataclass(frozen=True)
class PruningRequest:
    """A single tool observation and the intent needed to prune it."""

    text: str
    query: str = ""
    tool_type: str = "auto"
    path: str | None = None
    recent_context: tuple[str, ...] = ()
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    request_id: str | None = None

    @property
    def lines(self) -> list[str]:
        return self.text.splitlines()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PruningRequest":
        budget_payload = payload.get("budget", {})
        budget = (
            budget_payload
            if isinstance(budget_payload, BudgetConfig)
            else BudgetConfig(**dict(budget_payload))
        )
        return cls(
            text=str(payload.get("text", "")),
            query=str(payload.get("query", "")),
            tool_type=str(payload.get("tool_type", "auto")),
            path=(None if payload.get("path") is None else str(payload.get("path"))),
            recent_context=tuple(str(item) for item in payload.get("recent_context", ())),
            budget=budget,
            metadata=dict(payload.get("metadata", {})),
            request_id=(
                None if payload.get("request_id") is None else str(payload.get("request_id"))
            ),
        )


@dataclass(frozen=True)
class LineScore:
    line_no: int
    score: float
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "line_no": self.line_no,
            "score": self.score,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class PruningResult:
    method: str
    original_line_count: int
    kept_line_numbers: tuple[int, ...]
    pruned_text: str
    line_scores: tuple[LineScore, ...] = ()
    latency_ms: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    request_id: str | None = None

    @property
    def kept_line_count(self) -> int:
        return len(self.kept_line_numbers)

    @property
    def retention_ratio(self) -> float:
        if self.original_line_count == 0:
            return 1.0
        return self.kept_line_count / self.original_line_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "request_id": self.request_id,
            "original_line_count": self.original_line_count,
            "kept_line_count": self.kept_line_count,
            "retention_ratio": self.retention_ratio,
            "kept_line_numbers": list(self.kept_line_numbers),
            "pruned_text": self.pruned_text,
            "line_scores": [score.to_dict() for score in self.line_scores],
            "latency_ms": self.latency_ms,
            "metadata": dict(self.metadata),
        }


class Pruner(Protocol):
    name: str

    def prune(self, request: PruningRequest) -> PruningResult:
        """Prune one tool observation without updating model parameters."""


def coerce_line_scores(
    scores: Sequence[float],
    reasons: Mapping[int, Sequence[str]] | None = None,
) -> tuple[LineScore, ...]:
    reason_map = reasons or {}
    return tuple(
        LineScore(
            line_no=index,
            score=float(score),
            reasons=tuple(reason_map.get(index, ())),
        )
        for index, score in enumerate(scores, start=1)
    )
