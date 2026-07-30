from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from posterior_pruning.candidates import (
    CandidateConfig,
    build_blocks,
    deletion_order,
    render_kept_lines,
)
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
class GreedyBlocksConfig:
    max_evaluations: int = 6
    acceptance: AcceptanceConfig = field(default_factory=AcceptanceConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)

    def __post_init__(self) -> None:
        if self.max_evaluations < 1:
            raise ValueError("max_evaluations must be positive")


class GreedyBlocksMethod:
    """Greedily remove low-utility blocks, verifying every accepted deletion."""

    name = "greedy_blocks"

    def __init__(self, scorer: ActionScorer, config: GreedyBlocksConfig | None = None):
        self.scorer = scorer
        self.config = config or GreedyBlocksConfig()

    def prune(self, request: PosteriorPruningRequest) -> PosteriorPruningResult:
        started_at = time.perf_counter()
        lines = request.observation.splitlines()
        if request.keep_ratio >= 1.0 or len(lines) < 2:
            return unchanged_result(self.name, request, status="skipped", started_at=started_at)
        blocks = build_blocks(
            request.observation,
            next_action=request.next_action,
            query=request.query,
            config=self.config.candidates,
        )
        order = deletion_order(blocks)
        if not order:
            return unchanged_result(
                self.name,
                request,
                status="skipped",
                started_at=started_at,
                diagnostics={"reason": "all-blocks-protected"},
            )

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
        kept = set(range(1, len(lines) + 1))
        selected_score = full_score
        evaluated = 0
        accepted_deletions: list[tuple[int, int]] = []
        tested: list[dict[str, float | int | bool]] = []
        minimum_lines = max(1, math.ceil(len(lines) * request.keep_ratio))
        for block in order:
            if evaluated >= self.config.max_evaluations:
                break
            removable = tuple(line_no for line_no in block.line_numbers if line_no in kept)
            max_removable = len(kept) - minimum_lines
            if not removable or max_removable <= 0:
                continue
            removable = removable[-max_removable:]
            proposed_kept = kept - set(removable)
            candidate_text = render_kept_lines(lines, proposed_kept)
            forward_count += 1
            try:
                score = score_observation(self.scorer, request, candidate_text)
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
                    diagnostics={
                        "accepted_deletions_before_error": accepted_deletions,
                        "tested": tested,
                    },
                )
            prompt_tokens += score.prompt_tokens
            evaluated += 1
            accepted, drop = is_acceptable(full_score, score, self.config.acceptance)
            tested.append(
                {
                    "start_line": removable[0],
                    "end_line": removable[-1],
                    "utility": block.utility,
                    "mean_logprob_drop": drop,
                    "accepted": accepted,
                }
            )
            if accepted:
                kept = proposed_kept
                selected_score = score
                accepted_deletions.append((removable[0], removable[-1]))

        if len(kept) == len(lines):
            return result_from_selection(
                method=self.name,
                request=request,
                selected_text=request.observation,
                kept_line_numbers=tuple(sorted(kept)),
                status="rejected" if evaluated else "skipped",
                started_at=started_at,
                full_score=full_score,
                selected_score=full_score,
                model_forward_count=forward_count,
                candidates_evaluated=evaluated,
                scoring_prompt_tokens=prompt_tokens,
                diagnostics={"tested": tested},
            )
        selected_text = render_kept_lines(lines, kept)
        if not reduces_observation_tokens(request, selected_text):
            return result_from_selection(
                method=self.name,
                request=request,
                selected_text=request.observation,
                kept_line_numbers=tuple(range(1, len(lines) + 1)),
                status="rejected",
                started_at=started_at,
                full_score=full_score,
                selected_score=full_score,
                model_forward_count=forward_count,
                candidates_evaluated=evaluated,
                scoring_prompt_tokens=prompt_tokens,
                diagnostics={
                    "accepted_deletions": accepted_deletions,
                    "tested": tested,
                    "minimum_lines": minimum_lines,
                    "reason": "final-candidate-did-not-reduce-tokens",
                },
            )
        return result_from_selection(
            method=self.name,
            request=request,
            selected_text=selected_text,
            kept_line_numbers=tuple(sorted(kept)),
            status="accepted",
            started_at=started_at,
            full_score=full_score,
            selected_score=selected_score,
            model_forward_count=forward_count,
            candidates_evaluated=evaluated,
            scoring_prompt_tokens=prompt_tokens,
            diagnostics={
                "accepted_deletions": accepted_deletions,
                "tested": tested,
                "minimum_lines": minimum_lines,
            },
        )
