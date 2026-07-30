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
class BudgetSearchConfig:
    ratios: tuple[float, ...] = (0.25, 0.4, 0.6, 0.8)
    max_candidates: int = 4
    acceptance: AcceptanceConfig = field(default_factory=AcceptanceConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)

    def __post_init__(self) -> None:
        if not self.ratios or any(not 0.0 < ratio < 1.0 for ratio in self.ratios):
            raise ValueError("ratios must contain values in (0, 1)")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be positive")


class BudgetSearchMethod:
    """Find the smallest tested budget that preserves the recorded action."""

    name = "budget_search"

    def __init__(self, scorer: ActionScorer, config: BudgetSearchConfig | None = None):
        self.scorer = scorer
        self.config = config or BudgetSearchConfig()

    def _ratios(self, minimum: float) -> list[float]:
        values = {minimum}
        values.update(ratio for ratio in self.config.ratios if ratio >= minimum)
        return sorted(ratio for ratio in values if ratio < 1.0)[: self.config.max_candidates]

    def prune(self, request: PosteriorPruningRequest) -> PosteriorPruningResult:
        started_at = time.perf_counter()
        if request.keep_ratio >= 1.0:
            return unchanged_result(self.name, request, status="skipped", started_at=started_at)
        forward_count = 1
        prompt_tokens = 0
        try:
            full_score = score_observation(self.scorer, request, request.observation)
            prompt_tokens += full_score.prompt_tokens
        except Exception as exc:
            return scoring_error_result(
                self.name,
                request,
                started_at=started_at,
                error=exc,
                model_forward_count=forward_count,
                scoring_prompt_tokens=prompt_tokens,
            )
        evaluated = 0
        tested: list[dict[str, float | bool]] = []
        seen_text: set[str] = set()
        selected = None
        selected_score = None
        for ratio in self._ratios(request.keep_ratio):
            candidate = candidate_for_ratio(
                request.observation,
                next_action=request.next_action,
                query=request.query,
                keep_ratio=ratio,
                config=self.config.candidates,
            )
            if (
                candidate.text == request.observation
                or candidate.text in seen_text
                or not reduces_observation_tokens(request, candidate.text)
            ):
                continue
            seen_text.add(candidate.text)
            forward_count += 1
            try:
                score = score_observation(self.scorer, request, candidate.text)
            except Exception as exc:
                return scoring_error_result(
                    self.name,
                    request,
                    started_at=started_at,
                    error=exc,
                    model_forward_count=forward_count,
                    scoring_prompt_tokens=prompt_tokens,
                    candidates_evaluated=evaluated,
                    full_score=full_score,
                    diagnostics={"tested": tested},
                )
            prompt_tokens += score.prompt_tokens
            evaluated += 1
            accepted, drop = is_acceptable(full_score, score, self.config.acceptance)
            tested.append(
                {
                    "requested_ratio": ratio,
                    "actual_ratio": candidate.keep_ratio,
                    "mean_logprob": score.mean_logprob,
                    "mean_logprob_drop": drop,
                    "accepted": accepted,
                }
            )
            if accepted:
                selected = candidate
                selected_score = score
                break
        if selected is None or selected_score is None:
            return result_from_selection(
                method=self.name,
                request=request,
                selected_text=request.observation,
                kept_line_numbers=tuple(range(1, len(request.observation.splitlines()) + 1)),
                status="rejected" if evaluated else "skipped",
                started_at=started_at,
                full_score=full_score,
                selected_score=full_score,
                model_forward_count=forward_count,
                candidates_evaluated=evaluated,
                scoring_prompt_tokens=prompt_tokens,
                diagnostics={
                    "tested": tested,
                    "max_mean_logprob_drop": self.config.acceptance.max_mean_logprob_drop,
                },
            )
        return result_from_selection(
            method=self.name,
            request=request,
            selected_text=selected.text,
            kept_line_numbers=selected.kept_line_numbers,
            status="accepted",
            started_at=started_at,
            full_score=full_score,
            selected_score=selected_score,
            model_forward_count=forward_count,
            candidates_evaluated=evaluated,
            scoring_prompt_tokens=prompt_tokens,
            diagnostics={
                "selected_ratio": selected.keep_ratio,
                "tested": tested,
                "max_mean_logprob_drop": self.config.acceptance.max_mean_logprob_drop,
            },
        )
