from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from posterior_pruning.candidates import estimate_tokens


@dataclass(frozen=True)
class PosteriorClientConfig:
    url: str
    keep_ratio: float = 0.5
    timeout: float = 180.0
    min_chars: int = 500

    def __post_init__(self) -> None:
        if not self.url:
            raise ValueError("posterior URL must not be empty")
        if not 0.0 < self.keep_ratio <= 1.0:
            raise ValueError("posterior keep_ratio must be in (0, 1]")
        if self.timeout <= 0:
            raise ValueError("posterior timeout must be positive")
        if self.min_chars < 0:
            raise ValueError("posterior min_chars must be non-negative")

    @classmethod
    def from_env(cls) -> PosteriorClientConfig | None:
        url = os.getenv("POSTERIOR_PRUNER_URL", "").strip()
        if not url:
            return None
        return cls(
            url=url,
            keep_ratio=float(os.getenv("POSTERIOR_KEEP_RATIO", "0.5")),
            timeout=float(os.getenv("POSTERIOR_PRUNER_TIMEOUT", "180")),
            min_chars=int(os.getenv("POSTERIOR_MIN_CHARS", "500")),
        )

    @property
    def endpoint(self) -> str:
        url = self.url.rstrip("/")
        if url.endswith("/prune-post-action"):
            return url
        return f"{url}/prune-post-action"


class PosteriorClient:
    def __init__(self, config: PosteriorClientConfig):
        self.config = config

    def prune(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        observation_index: int,
        next_action: str,
        query: str = "",
        request_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        observation = messages[observation_index].get("content", "")
        if len(observation) < self.config.min_chars:
            token_count = estimate_tokens(observation)
            line_count = len(observation.splitlines())
            return {
                "status": "skipped",
                "method": "client",
                "pruned_response": observation,
                "original_line_count": line_count,
                "kept_line_count": line_count,
                "original_estimated_tokens": token_count,
                "kept_estimated_tokens": token_count,
                "retention_ratio": 1.0,
                "model_forward_count": 0,
                "candidates_evaluated": 0,
                "diagnostics": {
                    "reason": "below-client-min-chars",
                    "min_chars": self.config.min_chars,
                },
            }
        payload = {
            "messages": [
                {
                    "role": str(message.get("role", "")),
                    "content": str(message.get("content", "")),
                }
                for message in messages
            ],
            "observation_index": observation_index,
            "next_action": next_action,
            "keep_ratio": self.config.keep_ratio,
            "query": query,
            "request_id": request_id,
            "metadata": dict(metadata or {}),
        }
        request = urllib.request.Request(
            self.config.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"posterior service returned HTTP {exc.code}: {body[:500]}") from exc
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"posterior service request failed: {exc}") from exc
        if not isinstance(result, dict):
            raise RuntimeError("posterior service response must be a JSON object")
        pruned = result.get("pruned_response")
        if not isinstance(pruned, str):
            raise RuntimeError("posterior service response has no pruned_response string")
        return result
