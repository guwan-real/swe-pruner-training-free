from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


def _optional_string(value: Mapping[str, Any], name: str) -> str:
    result = value.get(name, "")
    if result is None:
        return ""
    if not isinstance(result, str):
        raise ValueError(f"{name} must be a string")
    return result


@dataclass(frozen=True)
class PruningRequest:
    """Compatible superset of the official SWE-Pruner ``POST /prune`` request."""

    query: str
    code: str
    threshold: float = 0.5
    command: str = ""
    path: str = ""
    task: str = ""
    recent_context: str = ""
    request_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise ValueError("query must be a string")
        if not isinstance(self.code, str):
            raise ValueError("code must be a string")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if self.request_id is not None and not self.request_id:
            raise ValueError("request_id must not be empty")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PruningRequest:
        query = value.get("query")
        code = value.get("code")
        threshold = value.get("threshold", 0.5)
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if not isinstance(code, str):
            raise ValueError("code must be a string")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("threshold must be numeric")
        request_id = value.get("request_id")
        if request_id is not None and not isinstance(request_id, str):
            raise ValueError("request_id must be a string or null")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be an object")
        recent_context = value.get("recent_context", "")
        if isinstance(recent_context, list):
            if not all(isinstance(item, str) for item in recent_context):
                raise ValueError("recent_context list items must be strings")
            recent_context = "\n".join(recent_context)
        if recent_context is None:
            recent_context = ""
        if not isinstance(recent_context, str):
            raise ValueError("recent_context must be a string or string array")
        return cls(
            query=query,
            code=code,
            threshold=float(threshold),
            command=_optional_string(value, "command"),
            path=_optional_string(value, "path"),
            task=_optional_string(value, "task"),
            recent_context=recent_context,
            request_id=request_id,
            metadata=dict(metadata),
        )

    @property
    def intent_text(self) -> str:
        """Return bounded, inference-time intent text without generating a goal hint."""

        parts = (self.query, self.command, self.path, self.task, self.recent_context[-4000:])
        return "\n".join(part.strip() for part in parts if part.strip())


@dataclass(frozen=True)
class PruningResult:
    """Response with the official fields plus auditable zero-forward diagnostics."""

    pruned_code: str
    origin_token_cnt: int
    left_token_cnt: int
    model_input_token_cnt: int
    method: str
    status: str
    original_line_count: int
    kept_line_count: int
    retention_ratio: float
    latency_ms: float
    model_forward_count: int = 0
    llm_token_count: int = 0
    raw_id: str | None = None
    recovery_url: str | None = None
    kept_line_numbers: tuple[int, ...] = ()
    score: tuple[float, ...] = ()
    error: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model_input_token_cnt != 0:
            raise ValueError("zero-forward pruning must report model_input_token_cnt=0")
        if self.model_forward_count != 0:
            raise ValueError("zero-forward pruning must report model_forward_count=0")
        if self.llm_token_count != 0:
            raise ValueError("zero-forward pruning must report llm_token_count=0")
        if self.origin_token_cnt < 0 or self.left_token_cnt < 0:
            raise ValueError("token counts must be non-negative")
        if not 0.0 <= self.retention_ratio <= 1.0:
            raise ValueError("retention_ratio must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
