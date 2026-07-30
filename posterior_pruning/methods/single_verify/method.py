from __future__ import annotations

import time
from dataclasses import dataclass, field

from posterior_pruning.candidates import CandidateConfig, candidate_for_ratio
from posterior_pruning.methods.common import (
    AcceptanceConfig,
    is_acceptable,
    reduces_observation_tokens,
    result_from_selection,
    score_observation,
    scoring_error_result,
    unchanged_result,
)
from posterior_pruning.protocol import PosteriorPruningRequest, PosteriorPruningResult
from posterior_pruning.scoring import ActionScorer


@dataclass(frozen=True)
class SingleVerifyConfig:
    acceptance: AcceptanceConfig = field(default_factory=AcceptanceConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)


class SingleVerifyMethod:
    """Verify one cheap action-conditioned candidate against the full context."""

    name = "single_verify"

    def __init__(self, scorer: ActionScorer, config: SingleVerifyConfig | None = None):
        self.scorer = scorer
        self.config = config or SingleVerifyConfig()

    def prune(self, request: PosteriorPruningRequest) -> PosteriorPruningResult:
        started_at = time.perf_counter()
        if request.keep_ratio >= 1.0:
            return unchanged_result(self.name, request, status="skipped", started_at=started_at)
        candidate = candidate_for_ratio(
            request.observation,
            next_action=request.next_action,
            query=request.query,
            keep_ratio=request.keep_ratio,
            config=self.config.candidates,
        )
        if candidate.text == request.observation or not reduces_observation_tokens(
            request, candidate.text
        ):
            return unchanged_result(
                self.name,
                request,
                status="skipped",
                started_at=started_at,
                diagnostics={"reason": "candidate-did-not-reduce-tokens"},
            )
        forward_count = 0
        prompt_tokens = 0
        full_score = None
        try:
            forward_count += 1
            full_score = score_observation(self.scorer, request, request.observation)
            prompt_tokens += full_score.prompt_tokens
            forward_count += 1
            candidate_score = score_observation(self.scorer, request, candidate.text)
            prompt_tokens += candidate_score.prompt_tokens
        except Exception as exc:
            return scoring_error_result(
                self.name,
                request,
                started_at=started_at,
                error=exc,
                model_forward_count=forward_count,
                scoring_prompt_tokens=prompt_tokens,
                candidates_evaluated=max(0, forward_count - 1),
                full_score=full_score,
            )
        accepted, drop = is_acceptable(
            full_score,
            candidate_score,
            self.config.acceptance,
        )
        return result_from_selection(
            method=self.name,
            request=request,
            selected_text=candidate.text if accepted else request.observation,
            kept_line_numbers=(
                candidate.kept_line_numbers
                if accepted
                else tuple(range(1, len(request.observation.splitlines()) + 1))
            ),
            status="accepted" if accepted else "rejected",
            started_at=started_at,
            full_score=full_score,
            selected_score=candidate_score if accepted else full_score,
            model_forward_count=forward_count,
            candidates_evaluated=1,
            scoring_prompt_tokens=prompt_tokens,
            diagnostics={
                "candidate_ratio": candidate.keep_ratio,
                "candidate_mean_logprob": candidate_score.mean_logprob,
                "tested_mean_logprob_drop": drop,
                "max_mean_logprob_drop": self.config.acceptance.max_mean_logprob_drop,
            },
        )
