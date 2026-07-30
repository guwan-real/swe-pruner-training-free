from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from posterior_pruning.candidates import CandidateConfig, build_blocks, render_kept_lines
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
class BlockInfluenceConfig:
    max_block_evaluations: int = 6
    acceptance: AcceptanceConfig = field(default_factory=AcceptanceConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)

    def __post_init__(self) -> None:
        if self.max_block_evaluations < 1:
            raise ValueError("max_block_evaluations must be positive")


class BlockInfluenceMethod:
    """Small-sample leave-one-block-out posterior influence oracle.

    Every eligible block is deleted independently to estimate its harm to the
    fixed next action.  The combined keep set is scored once more and is used
    only when that final counterfactual also passes the acceptance gate.
    """

    name = "block_influence"

    def __init__(self, scorer: ActionScorer, config: BlockInfluenceConfig | None = None):
        self.scorer = scorer
        self.config = config or BlockInfluenceConfig()

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
        eligible = [block for block in blocks if not block.protected][
            : self.config.max_block_evaluations
        ]
        if not eligible:
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
        all_lines = set(range(1, len(lines) + 1))
        influence: list[tuple[float, int, int, tuple[int, ...]]] = []
        tested: list[dict[str, float | int]] = []
        for block in eligible:
            candidate_text = render_kept_lines(lines, all_lines - set(block.line_numbers))
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
                    candidates_evaluated=len(tested),
                    full_score=full_score,
                    diagnostics={"block_influence": tested},
                )
            prompt_tokens += score.prompt_tokens
            harm = full_score.mean_logprob - score.mean_logprob
            influence.append((harm, block.start_line, block.end_line, block.line_numbers))
            tested.append(
                {
                    "start_line": block.start_line,
                    "end_line": block.end_line,
                    "mean_logprob_harm": harm,
                }
            )

        target_lines = max(1, math.ceil(len(lines) * request.keep_ratio))
        protected_lines = {
            line_no for block in blocks if block.protected for line_no in block.line_numbers
        }
        kept = set(protected_lines)
        for _, _, _, block_lines in sorted(influence, key=lambda item: (-item[0], item[1])):
            if len(kept) >= target_lines:
                break
            needed = target_lines - len(kept)
            kept.update(block_lines[:needed])
        if len(kept) < target_lines:
            unranked = [block for block in blocks if block.protected or block not in eligible]
            for block in sorted(unranked, key=lambda item: (-item.utility, item.start_line)):
                if len(kept) >= target_lines:
                    break
                needed = target_lines - len(kept)
                kept.update(block.line_numbers[:needed])
        candidate_text = render_kept_lines(lines, kept)
        if candidate_text == request.observation:
            return result_from_selection(
                method=self.name,
                request=request,
                selected_text=request.observation,
                kept_line_numbers=tuple(sorted(all_lines)),
                status="skipped",
                started_at=started_at,
                full_score=full_score,
                selected_score=full_score,
                model_forward_count=forward_count,
                candidates_evaluated=len(eligible),
                scoring_prompt_tokens=prompt_tokens,
                diagnostics={"block_influence": tested},
            )
        forward_count += 1
        try:
            final_score = score_observation(self.scorer, request, candidate_text)
        except Exception as exc:
            return scoring_error_result(
                self.name,
                request,
                started_at=started_at,
                error=exc,
                model_forward_count=forward_count,
                scoring_prompt_tokens=prompt_tokens,
                candidates_evaluated=len(tested),
                full_score=full_score,
                diagnostics={"block_influence": tested},
            )
        prompt_tokens += final_score.prompt_tokens
        accepted, drop = is_acceptable(full_score, final_score, self.config.acceptance)
        token_reduction = reduces_observation_tokens(request, candidate_text)
        accepted = accepted and token_reduction
        return result_from_selection(
            method=self.name,
            request=request,
            selected_text=candidate_text if accepted else request.observation,
            kept_line_numbers=(tuple(sorted(kept)) if accepted else tuple(sorted(all_lines))),
            status="accepted" if accepted else "rejected",
            started_at=started_at,
            full_score=full_score,
            selected_score=final_score if accepted else full_score,
            model_forward_count=forward_count,
            candidates_evaluated=1 + len(eligible),
            scoring_prompt_tokens=prompt_tokens,
            diagnostics={
                "block_influence": tested,
                "final_candidate_mean_logprob": final_score.mean_logprob,
                "final_mean_logprob_drop": drop,
                "final_candidate_reduces_tokens": token_reduction,
                "oracle_scope": "small-sample-online",
            },
        )
