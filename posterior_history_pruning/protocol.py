from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PosteriorHistoryConfig:
    """Safety-first configuration for delayed history compaction."""

    hot_observations: int = 2
    min_input_tokens: int = 1500
    min_savings_tokens: int = 256
    max_retention_ratio: float = 0.85
    block_max_lines: int = 16
    max_output_chars: int = 9000
    method: str = "adaptive"

    def __post_init__(self) -> None:
        if self.hot_observations < 1:
            raise ValueError("hot_observations must be at least 1")
        if self.min_input_tokens < 0:
            raise ValueError("min_input_tokens must be non-negative")
        if self.min_savings_tokens < 1:
            raise ValueError("min_savings_tokens must be positive")
        if not 0.0 < self.max_retention_ratio < 1.0:
            raise ValueError("max_retention_ratio must be in (0, 1)")
        if self.block_max_lines < 1:
            raise ValueError("block_max_lines must be positive")
        if self.max_output_chars < 1000:
            raise ValueError("max_output_chars must be at least 1000")
        if self.method not in {"safe", "adaptive"}:
            raise ValueError("method must be 'safe' or 'adaptive'")

    @classmethod
    def from_env(cls) -> PosteriorHistoryConfig | None:
        if os.getenv("POSTERIOR_HISTORY_ENABLED", "0") != "1":
            return None
        return cls(
            hot_observations=int(os.getenv("POSTERIOR_HOT_OBSERVATIONS", "2")),
            min_input_tokens=int(os.getenv("POSTERIOR_MIN_INPUT_TOKENS", "1500")),
            min_savings_tokens=int(os.getenv("POSTERIOR_MIN_SAVINGS_TOKENS", "256")),
            max_retention_ratio=float(os.getenv("POSTERIOR_MAX_RETENTION_RATIO", "0.85")),
            block_max_lines=int(os.getenv("POSTERIOR_BLOCK_MAX_LINES", "16")),
            max_output_chars=int(os.getenv("POSTERIOR_MAX_OUTPUT_CHARS", "9000")),
            method=os.getenv("POSTERIOR_HISTORY_METHOD", "adaptive").strip().lower(),
        )


@dataclass(frozen=True)
class PosteriorSignal:
    """Signals emitted by the agent's normal follow-up action.

    They are *not* produced by an extra inference request.  The hook derives
    them only after the model has already read the complete observation.
    """

    command: str = ""
    context_focus_question: str = ""
    response_content: str = ""

    @property
    def text(self) -> str:
        # The shell action is the strongest and most stable posterior signal.
        # Keep prose bounded: it can contain a long chain-of-thought-like
        # explanation that should not dominate lexical selection.
        return "\n".join(
            value.strip()
            for value in (self.context_focus_question, self.command, self.response_content[:800])
            if value and value.strip()
        )


@dataclass(frozen=True)
class CompactionResult:
    text: str
    status: str
    reason: str
    method: str
    output_kind: str
    origin_token_cnt: int
    left_token_cnt: int
    original_line_count: int
    kept_line_count: int
    retained_line_numbers: tuple[int, ...] = ()

    @property
    def retention_ratio(self) -> float:
        return self.left_token_cnt / self.origin_token_cnt if self.origin_token_cnt else 1.0
