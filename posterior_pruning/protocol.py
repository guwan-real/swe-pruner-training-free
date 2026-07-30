from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, index: int) -> ChatMessage:
        role = value.get("role")
        content = value.get("content")
        if not isinstance(role, str) or not role:
            raise ValueError(f"messages[{index}].role must be a non-empty string")
        if not isinstance(content, str):
            raise ValueError(f"messages[{index}].content must be a string")
        return cls(role=role, content=content)

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class PosteriorPruningRequest:
    """The post-action contract used by the isolated posterior service.

    ``messages`` contains the exact history used to generate ``next_action``.
    ``observation_index`` points at the full tool observation within that
    history.  The service may replace that one message while scoring the fixed
    action, but it never generates or changes the action.
    """

    messages: tuple[ChatMessage, ...]
    observation_index: int
    next_action: str
    keep_ratio: float = 0.5
    query: str = ""
    request_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("messages must not be empty")
        if not 0 <= self.observation_index < len(self.messages):
            raise ValueError("observation_index is outside messages")
        if self.messages[self.observation_index].role != "user":
            raise ValueError("the observation message must have role='user'")
        if not self.next_action:
            raise ValueError("next_action must not be empty")
        if not 0.0 < self.keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1]")

    @property
    def observation(self) -> str:
        return self.messages[self.observation_index].content

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PosteriorPruningRequest:
        raw_messages = value.get("messages")
        if not isinstance(raw_messages, list):
            raise ValueError("messages must be a JSON array")
        messages = tuple(
            ChatMessage.from_mapping(message, index=index)
            for index, message in enumerate(raw_messages)
            if isinstance(message, Mapping)
        )
        if len(messages) != len(raw_messages):
            raise ValueError("every messages item must be an object")

        raw_index = value.get("observation_index")
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise ValueError("observation_index must be an integer")
        next_action = value.get("next_action")
        if not isinstance(next_action, str):
            raise ValueError("next_action must be a string")
        keep_ratio = value.get("keep_ratio", 0.5)
        if isinstance(keep_ratio, bool) or not isinstance(keep_ratio, (int, float)):
            raise ValueError("keep_ratio must be numeric")
        query = value.get("query", "")
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        request_id = value.get("request_id")
        if request_id is not None and not isinstance(request_id, str):
            raise ValueError("request_id must be a string or null")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be an object")
        return cls(
            messages=messages,
            observation_index=raw_index,
            next_action=next_action,
            keep_ratio=float(keep_ratio),
            query=query,
            request_id=request_id,
            metadata=dict(metadata),
        )

    def messages_with_observation(self, observation: str) -> list[dict[str, str]]:
        messages = [message.to_dict() for message in self.messages]
        messages[self.observation_index] = {
            "role": messages[self.observation_index]["role"],
            "content": observation,
        }
        return messages


@dataclass(frozen=True)
class PosteriorPruningResult:
    method: str
    status: str
    pruned_response: str
    original_line_count: int
    kept_line_count: int
    original_estimated_tokens: int
    kept_estimated_tokens: int
    retention_ratio: float
    kept_line_numbers: tuple[int, ...] = ()
    full_action_mean_logprob: float | None = None
    selected_action_mean_logprob: float | None = None
    mean_logprob_drop: float | None = None
    action_token_count: int = 0
    model_forward_count: int = 0
    scoring_prompt_tokens: int = 0
    candidates_evaluated: int = 0
    latency_ms: float = 0.0
    error: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
