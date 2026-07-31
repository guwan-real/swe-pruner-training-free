from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping

from zero_forward_pruning.text import estimate_tokens


@dataclass(frozen=True)
class ZeroForwardClientConfig:
    url: str
    threshold: float = 0.5
    timeout: float = 5.0
    min_chars: int = 1000
    recovery_max_chars: int = 3000

    def __post_init__(self) -> None:
        if not self.url:
            raise ValueError("zero-forward URL must not be empty")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.min_chars < 0:
            raise ValueError("min_chars must be non-negative")
        if self.recovery_max_chars < 256:
            raise ValueError("recovery_max_chars must be at least 256")

    @classmethod
    def from_env(cls) -> ZeroForwardClientConfig | None:
        url = os.getenv("ZERO_FORWARD_PRUNER_URL", "").strip()
        if not url:
            return None
        return cls(
            url=url,
            threshold=float(os.getenv("ZERO_FORWARD_THRESHOLD", "0.5")),
            timeout=float(os.getenv("ZERO_FORWARD_TIMEOUT", "5")),
            min_chars=int(os.getenv("ZERO_FORWARD_MIN_CHARS", "1000")),
            recovery_max_chars=int(os.getenv("ZERO_FORWARD_RECOVERY_MAX_CHARS", "3000")),
        )

    @property
    def endpoint(self) -> str:
        base = self.url.rstrip("/")
        return base if base.endswith("/prune") else f"{base}/prune"


def _skipped(code: str, reason: str) -> dict[str, Any]:
    line_count = len(code.splitlines())
    token_count = estimate_tokens(code)
    return {
        "pruned_code": code,
        "status": "skipped",
        "method": "client",
        "origin_token_cnt": token_count,
        "left_token_cnt": token_count,
        "model_input_token_cnt": 0,
        "model_forward_count": 0,
        "llm_token_count": 0,
        "original_line_count": line_count,
        "kept_line_count": line_count,
        "retention_ratio": 1.0,
        "diagnostics": {"reason": reason},
    }


class ZeroForwardClient:
    def __init__(self, config: ZeroForwardClientConfig):
        self.config = config

    def prune(
        self,
        *,
        code: str,
        query: str,
        command: str = "",
        path: str = "",
        task: str = "",
        recent_context: str = "",
        request_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if len(code) < self.config.min_chars:
            return _skipped(code, "below-client-min-chars")
        payload = {
            "query": query,
            "code": code,
            "threshold": self.config.threshold,
            "command": command,
            "path": path,
            "task": task,
            "recent_context": recent_context,
            "request_id": request_id,
            "metadata": dict(metadata or {}),
        }
        request = urllib.request.Request(
            self.config.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"zero-forward service returned HTTP {exc.code}: {body[:500]}"
            ) from exc
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"zero-forward service request failed: {exc}") from exc
        if not isinstance(result, dict):
            raise RuntimeError("zero-forward service response must be a JSON object")
        if not isinstance(result.get("pruned_code"), str):
            raise RuntimeError("zero-forward service response has no pruned_code string")
        invariants = {
            "model_input_token_cnt": 0,
            "model_forward_count": 0,
            "llm_token_count": 0,
        }
        for name, expected in invariants.items():
            if result.get(name) != expected:
                raise RuntimeError(f"zero-forward invariant violated: {name}={result.get(name)!r}")
        return result
