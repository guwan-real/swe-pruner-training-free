from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence


class ScoringError(RuntimeError):
    pass


@dataclass(frozen=True)
class ActionScore:
    sum_logprob: float
    mean_logprob: float
    target_tokens: int
    prompt_tokens: int


class ActionScorer(Protocol):
    name: str

    def score(self, messages: Sequence[Mapping[str, str]], next_action: str) -> ActionScore:
        """Score the fixed action under an exact chat history."""


Transport = Callable[[str, str, Mapping[str, Any] | None, float], Mapping[str, Any]]


@dataclass(frozen=True)
class VLLMScorerConfig:
    api_base: str = "http://127.0.0.1:8015/v1"
    model: str = ""
    api_key: str = "EMPTY"
    timeout: float = 180.0
    chat_template_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.api_base:
            raise ValueError("api_base must not be empty")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")


def _default_transport(
    method: str,
    url: str,
    payload: Mapping[str, Any] | None,
    timeout: float,
    *,
    api_key: str,
) -> Mapping[str, Any]:
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ScoringError(f"{method} {url} returned HTTP {exc.code}: {body[:500]}") from exc
    except (OSError, ValueError) as exc:
        raise ScoringError(f"{method} {url} failed: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ScoringError(f"{method} {url} did not return a JSON object")
    return value


class VLLMActionScorer:
    """Fixed-action likelihood using vLLM prompt log probabilities.

    A cheap ``/tokenize`` request locates the assistant-generation boundary.
    One ``/chat/completions`` forward then returns prompt log probabilities for
    the history plus the already-generated action.  No weights are trained or
    updated.
    """

    name = "vllm-chat-prompt-logprobs"

    def __init__(
        self,
        config: VLLMScorerConfig | None = None,
        *,
        transport: Transport | None = None,
    ) -> None:
        self.config = config or VLLMScorerConfig()
        self._transport = transport
        self._model = self.config.model

    @property
    def api_base(self) -> str:
        return self.config.api_base.rstrip("/")

    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        url = f"{self.api_base}/{path.lstrip('/')}"
        if self._transport is not None:
            return self._transport(method, url, payload, self.config.timeout)
        return _default_transport(
            method,
            url,
            payload,
            self.config.timeout,
            api_key=self.config.api_key,
        )

    def resolve_model(self) -> str:
        if self._model:
            return self._model
        response = self._request("GET", "models")
        models = response.get("data")
        if not isinstance(models, list) or not models:
            raise ScoringError("vLLM /models returned no models")
        identifiers = [
            item.get("id")
            for item in models
            if isinstance(item, Mapping) and isinstance(item.get("id"), str)
        ]
        if not identifiers:
            raise ScoringError("vLLM /models response has no model id")
        preferred = [item for item in identifiers if "qwen3.5" in item.lower()]
        self._model = preferred[0] if preferred else identifiers[0]
        return self._model

    @staticmethod
    def _clean_messages(
        messages: Sequence[Mapping[str, str]],
    ) -> list[dict[str, str]]:
        cleaned: list[dict[str, str]] = []
        for index, message in enumerate(messages):
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise ScoringError(f"message {index} must contain string role/content")
            cleaned.append({"role": role, "content": content})
        return cleaned

    def _tokenize_prefix(self, messages: list[dict[str, str]]) -> list[int]:
        payload: dict[str, Any] = {
            "model": self.resolve_model(),
            "messages": messages,
            "add_generation_prompt": True,
            "continue_final_message": False,
            "add_special_tokens": False,
        }
        if self.config.chat_template_kwargs:
            payload["chat_template_kwargs"] = dict(self.config.chat_template_kwargs)
        response = self._request("POST", "tokenize", payload)
        tokens = response.get("tokens")
        if not isinstance(tokens, list) or not all(
            isinstance(token, int) and not isinstance(token, bool) for token in tokens
        ):
            raise ScoringError("vLLM /tokenize response has no integer tokens array")
        return tokens

    @staticmethod
    def _actual_logprob(entry: Any, token_id: int, *, position: int) -> float:
        if not isinstance(entry, Mapping):
            raise ScoringError(f"missing prompt logprob at target position {position}")
        value = entry.get(str(token_id), entry.get(token_id))
        if value is None:
            raise ScoringError(
                f"prompt logprobs at position {position} do not contain actual token {token_id}"
            )
        if isinstance(value, Mapping):
            value = value.get("logprob")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ScoringError(f"invalid prompt logprob at target position {position}")
        return float(value)

    def score(
        self,
        messages: Sequence[Mapping[str, str]],
        next_action: str,
    ) -> ActionScore:
        if not next_action:
            raise ScoringError("next_action must not be empty")
        history = self._clean_messages(messages)
        prefix_tokens = self._tokenize_prefix(history)
        full_messages = [*history, {"role": "assistant", "content": next_action}]
        payload: dict[str, Any] = {
            "model": self.resolve_model(),
            "messages": full_messages,
            "add_generation_prompt": False,
            "continue_final_message": False,
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
            "prompt_logprobs": 1,
            "return_token_ids": True,
        }
        if self.config.chat_template_kwargs:
            payload["chat_template_kwargs"] = dict(self.config.chat_template_kwargs)
        response = self._request("POST", "chat/completions", payload)
        prompt_tokens = response.get("prompt_token_ids")
        prompt_logprobs = response.get("prompt_logprobs")
        if not isinstance(prompt_tokens, list) or not all(
            isinstance(token, int) and not isinstance(token, bool) for token in prompt_tokens
        ):
            raise ScoringError(
                "vLLM response omitted prompt_token_ids; the server must support "
                "return_token_ids=true"
            )
        if not isinstance(prompt_logprobs, list):
            raise ScoringError(
                "vLLM response omitted prompt_logprobs; start a compatible vLLM server"
            )
        if len(prompt_tokens) != len(prompt_logprobs):
            raise ScoringError("vLLM prompt token/logprob lengths differ")
        if prompt_tokens[: len(prefix_tokens)] != prefix_tokens:
            raise ScoringError(
                "chat-template prefix mismatch between /tokenize and /chat/completions; "
                "use the same model and chat_template_kwargs for agent and scorer"
            )
        target_start = len(prefix_tokens)
        if target_start >= len(prompt_tokens):
            raise ScoringError("the fixed action produced zero scoreable prompt tokens")
        values = [
            self._actual_logprob(prompt_logprobs[index], prompt_tokens[index], position=index)
            for index in range(target_start, len(prompt_tokens))
        ]
        total = sum(values)
        return ActionScore(
            sum_logprob=total,
            mean_logprob=total / len(values),
            target_tokens=len(values),
            prompt_tokens=len(prompt_tokens),
        )
