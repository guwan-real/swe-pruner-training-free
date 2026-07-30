from __future__ import annotations

from posterior_pruning.candidates import CandidateConfig
from posterior_pruning.methods.block_influence import (
    BlockInfluenceConfig,
    BlockInfluenceMethod,
)
from posterior_pruning.methods.budget_search import BudgetSearchConfig, BudgetSearchMethod
from posterior_pruning.methods.common import AcceptanceConfig
from posterior_pruning.methods.greedy_blocks import GreedyBlocksConfig, GreedyBlocksMethod
from posterior_pruning.methods.single_verify import SingleVerifyConfig, SingleVerifyMethod
from posterior_pruning.protocol import ChatMessage, PosteriorPruningRequest
from posterior_pruning.scoring import ActionScore


class RequiredEvidenceScorer:
    name = "required-evidence"

    def __init__(self) -> None:
        self.observations: list[str] = []

    def score(self, messages, next_action: str) -> ActionScore:
        observation = messages[3]["content"]
        self.observations.append(observation)
        mean = 0.0 if "REQUIRED" in observation else -1.0
        return ActionScore(
            sum_logprob=mean * 2,
            mean_logprob=mean,
            target_tokens=2,
            prompt_tokens=10,
        )


class FailsOnSecondScore(RequiredEvidenceScorer):
    def score(self, messages, next_action: str) -> ActionScore:
        if self.observations:
            raise RuntimeError("candidate scoring failed")
        return super().score(messages, next_action)


def request(keep_ratio: float = 0.25) -> PosteriorPruningRequest:
    return PosteriorPruningRequest(
        messages=(
            ChatMessage("system", "system"),
            ChatMessage("user", "task"),
            ChatMessage("assistant", "```bash\ncat file\n```"),
            ChatMessage(
                "user",
                "\n".join(
                    (
                        (
                            "head alpha beta gamma delta epsilon zeta eta theta "
                            "iota kappa lambda mu nu xi omicron"
                        ),
                        (
                            "REQUIRED parser evidence exact source location alpha beta "
                            "gamma delta epsilon zeta eta theta iota kappa"
                        ),
                        (
                            "noise one two three four five six seven eight nine ten "
                            "eleven twelve thirteen fourteen fifteen sixteen"
                        ),
                        (
                            "tail alpha beta gamma delta epsilon zeta eta theta "
                            "iota kappa lambda mu nu xi omicron"
                        ),
                    )
                ),
            ),
        ),
        observation_index=3,
        next_action="```bash\nedit another_file\n```",
        keep_ratio=keep_ratio,
    )


NO_PROTECTION = CandidateConfig(
    block_max_lines=1,
    protect_errors=False,
    protect_diffs=False,
    protect_edge_lines=False,
)
STRICT = AcceptanceConfig(max_mean_logprob_drop=0.1)


def test_single_verify_falls_back_to_full_observation() -> None:
    scorer = RequiredEvidenceScorer()
    method = SingleVerifyMethod(
        scorer,
        SingleVerifyConfig(acceptance=STRICT, candidates=NO_PROTECTION),
    )

    result = method.prune(request())

    assert result.status == "rejected"
    assert result.pruned_response == request().observation
    assert result.model_forward_count == 2
    assert result.mean_logprob_drop == 0.0
    assert "REQUIRED" not in scorer.observations[1]


def test_single_verify_does_not_accept_marker_overhead_as_savings() -> None:
    value = request()
    short_request = PosteriorPruningRequest(
        messages=(
            *value.messages[:3],
            ChatMessage("user", "a\nb\nc\nd"),
        ),
        observation_index=3,
        next_action=value.next_action,
        keep_ratio=0.25,
    )
    method = SingleVerifyMethod(
        RequiredEvidenceScorer(),
        SingleVerifyConfig(acceptance=STRICT, candidates=NO_PROTECTION),
    )

    result = method.prune(short_request)

    assert result.status == "skipped"
    assert result.model_forward_count == 0
    assert result.retention_ratio == 1.0
    assert result.diagnostics["reason"] == "candidate-did-not-reduce-tokens"


def test_method_failure_preserves_observation_and_attempted_forward_cost() -> None:
    method = SingleVerifyMethod(
        FailsOnSecondScore(),
        SingleVerifyConfig(acceptance=STRICT, candidates=NO_PROTECTION),
    )

    result = method.prune(request())

    assert result.status == "error"
    assert result.pruned_response == request().observation
    assert result.model_forward_count == 2
    assert result.scoring_prompt_tokens == 10
    assert "candidate scoring failed" in result.error


def test_budget_search_chooses_first_posterior_safe_budget() -> None:
    method = BudgetSearchMethod(
        RequiredEvidenceScorer(),
        BudgetSearchConfig(
            ratios=(0.25, 0.5, 0.75),
            max_candidates=3,
            acceptance=STRICT,
            candidates=NO_PROTECTION,
        ),
    )

    result = method.prune(request())

    assert result.status == "accepted"
    assert "REQUIRED" in result.pruned_response
    assert result.retention_ratio < 1.0
    assert result.candidates_evaluated == 3
    assert result.model_forward_count == 4


def test_greedy_blocks_checks_each_deletion_against_full_posterior() -> None:
    method = GreedyBlocksMethod(
        RequiredEvidenceScorer(),
        GreedyBlocksConfig(
            max_evaluations=4,
            acceptance=STRICT,
            candidates=NO_PROTECTION,
        ),
    )

    result = method.prune(request())

    assert result.status == "accepted"
    assert "REQUIRED" in result.pruned_response
    assert "noise" not in result.pruned_response
    assert any(not item["accepted"] for item in result.diagnostics["tested"])
    assert result.model_forward_count <= 5


def test_block_influence_is_bounded_and_final_candidate_is_verified() -> None:
    method = BlockInfluenceMethod(
        RequiredEvidenceScorer(),
        BlockInfluenceConfig(
            max_block_evaluations=4,
            acceptance=STRICT,
            candidates=NO_PROTECTION,
        ),
    )

    result = method.prune(request())

    assert result.status == "accepted"
    assert "REQUIRED" in result.pruned_response
    assert result.model_forward_count == 6
    assert result.diagnostics["oracle_scope"] == "small-sample-online"
