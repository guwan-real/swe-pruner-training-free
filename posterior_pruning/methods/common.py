from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping

from posterior_pruning.candidates import estimate_tokens
from posterior_pruning.protocol import PosteriorPruningRequest, PosteriorPruningResult
from posterior_pruning.scoring import ActionScore, ActionScorer


@dataclass(frozen=True)
class AcceptanceConfig:
    max_mean_logprob_drop: float = 0.08

    def __post_init__(self) -> None:
        if self.max_mean_logprob_drop < 0:
            raise ValueError("max_mean_logprob_drop must be non-negative")


def score_observation(
    scorer: ActionScorer,
    request: PosteriorPruningRequest,
    observation: str,
) -> ActionScore:
    messages = request.messages_with_observation(observation)
    return scorer.score(messages, request.next_action)


def reduces_observation_tokens(request: PosteriorPruningRequest, candidate: str) -> bool:
    """Return whether the exact rendered candidate is shorter than the source."""

    return estimate_tokens(candidate) < estimate_tokens(request.observation)


def is_acceptable(
    full_score: ActionScore,
    candidate_score: ActionScore,
    config: AcceptanceConfig,
) -> tuple[bool, float]:
    drop = full_score.mean_logprob - candidate_score.mean_logprob
    return drop <= config.max_mean_logprob_drop, drop


def result_from_selection(
    *,
    method: str,
    request: PosteriorPruningRequest,
    selected_text: str,
    kept_line_numbers: tuple[int, ...],
    status: str,
    started_at: float,
    full_score: ActionScore | None = None,
    selected_score: ActionScore | None = None,
    model_forward_count: int = 0,
    candidates_evaluated: int = 0,
    scoring_prompt_tokens: int = 0,
    diagnostics: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> PosteriorPruningResult:
    original_lines = request.observation.splitlines()
    original_tokens = estimate_tokens(request.observation)
    # Count the exact text returned to the agent, including omission markers.
    # Those markers consume context and must not be hidden from the savings.
    kept_tokens = estimate_tokens(selected_text)
    retention = kept_tokens / original_tokens if original_tokens else 1.0
    mean_drop = None
    if full_score is not None and selected_score is not None:
        mean_drop = full_score.mean_logprob - selected_score.mean_logprob
    return PosteriorPruningResult(
        method=method,
        status=status,
        pruned_response=selected_text,
        original_line_count=len(original_lines),
        kept_line_count=len(kept_line_numbers),
        original_estimated_tokens=original_tokens,
        kept_estimated_tokens=kept_tokens,
        retention_ratio=retention,
        kept_line_numbers=kept_line_numbers,
        full_action_mean_logprob=(full_score.mean_logprob if full_score else None),
        selected_action_mean_logprob=(selected_score.mean_logprob if selected_score else None),
        mean_logprob_drop=mean_drop,
        action_token_count=(selected_score or full_score).target_tokens
        if (selected_score or full_score)
        else 0,
        model_forward_count=model_forward_count,
        scoring_prompt_tokens=scoring_prompt_tokens,
        candidates_evaluated=candidates_evaluated,
        latency_ms=(time.perf_counter() - started_at) * 1000,
        error=error,
        diagnostics=dict(diagnostics or {}),
    )


def unchanged_result(
    method: str,
    request: PosteriorPruningRequest,
    *,
    status: str,
    started_at: float,
    error: str | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> PosteriorPruningResult:
    lines = request.observation.splitlines()
    return result_from_selection(
        method=method,
        request=request,
        selected_text=request.observation,
        kept_line_numbers=tuple(range(1, len(lines) + 1)),
        status=status,
        started_at=started_at,
        error=error,
        diagnostics=diagnostics,
    )


def scoring_error_result(
    method: str,
    request: PosteriorPruningRequest,
    *,
    started_at: float,
    error: Exception,
    model_forward_count: int,
    scoring_prompt_tokens: int,
    candidates_evaluated: int = 0,
    full_score: ActionScore | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> PosteriorPruningResult:
    """Fail open while preserving attempted scorer cost."""

    lines = request.observation.splitlines()
    return result_from_selection(
        method=method,
        request=request,
        selected_text=request.observation,
        kept_line_numbers=tuple(range(1, len(lines) + 1)),
        status="error",
        started_at=started_at,
        full_score=full_score,
        selected_score=full_score,
        model_forward_count=model_forward_count,
        scoring_prompt_tokens=scoring_prompt_tokens,
        candidates_evaluated=candidates_evaluated,
        error=str(error),
        diagnostics=diagnostics,
    )
