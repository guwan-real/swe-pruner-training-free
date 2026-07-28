from __future__ import annotations

import pytest

from tasks.influence_oracle import (
    HFLogLikelihoodScorer,
    InfluenceOracleConfig,
    InfluenceOraclePruner,
    build_pruner,
)
from tf_pruning.protocol import BudgetConfig, PruningRequest


class DeterministicActionScorer:
    name = "deterministic-action-likelihood"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def log_likelihood(self, context: str, continuation: str) -> float:
        self.calls.append((context, continuation))
        return 10.0 if "CRITICAL evidence" in context else 0.0


class CustomObjective:
    name = "custom-objective"

    def target(self, request: PruningRequest) -> str:
        return str(request.metadata["gold_action"])

    def prompt(self, request: PruningRequest, observation: str) -> str:
        return f"CUSTOM OBJECTIVE\n{observation}\nACTION:\n"


def test_leave_one_block_out_ranks_next_action_likelihood_harm() -> None:
    text = "\n".join(
        (
            "noise alpha",
            "noise beta",
            "",
            "CRITICAL evidence",
            "because parser",
            "",
            "irrelevant tail",
        )
    )
    scorer = DeterministicActionScorer()
    request = PruningRequest(
        text=text,
        budget=BudgetConfig(
            keep_ratio=2 / 7,
            min_lines=2,
            no_prune_below=0,
            context_window=0,
        ),
        metadata={"gold_action": "open src/parser.py"},
        request_id="oracle-loo",
    )
    pruner = InfluenceOraclePruner(
        scorer=scorer,
        objective=CustomObjective(),
        config=InfluenceOracleConfig(
            strategy="leave_one_out",
            block_max_lines=8,
        ),
    )

    result = pruner.prune(request)

    assert result.kept_line_numbers == (4, 5)
    assert result.line_scores[3].score == 10.0
    assert result.line_scores[4].score == 10.0
    assert result.metadata["full_log_likelihood"] == 10.0
    assert result.metadata["evaluations"] == result.metadata["block_count"] + 1
    assert result.metadata["objective"] == "custom-objective"
    assert result.metadata["oracle_scope"] == "small-sample-offline"
    assert all(target == "open src/parser.py" for _, target in scorer.calls)
    assert all(context.startswith("CUSTOM OBJECTIVE") for context, _ in scorer.calls)


def test_hierarchical_greedy_refines_block_to_exact_line_budget() -> None:
    scorer = DeterministicActionScorer()
    request = PruningRequest(
        text="\n".join(
            (
                "noise one",
                "CRITICAL evidence",
                "noise two",
                "noise three",
            )
        ),
        budget=BudgetConfig(
            keep_ratio=0.5,
            min_lines=2,
            no_prune_below=0,
            context_window=0,
        ),
        metadata={"next_action": "edit parser"},
    )
    pruner = InfluenceOraclePruner(
        scorer=scorer,
        config=InfluenceOracleConfig(
            strategy="hierarchical_greedy",
            block_max_lines=8,
            max_evaluations=20,
        ),
    )

    result = pruner.prune(request)

    assert result.kept_line_count == 2
    assert 2 in result.kept_line_numbers
    assert result.kept_line_numbers == (2, 4)
    assert result.metadata["strategy"] == "hierarchical_greedy"
    assert result.metadata["evaluations"] == 8
    assert "greedy-kept" in result.line_scores[1].reasons
    assert any(reason.startswith("greedy-removed-step") for reason in result.line_scores[0].reasons)


def test_next_action_metadata_is_required_when_scoring() -> None:
    request = PruningRequest(
        text="one\ntwo",
        budget=BudgetConfig(
            keep_ratio=0.5,
            min_lines=1,
            no_prune_below=0,
        ),
    )
    with pytest.raises(ValueError, match="next_action"):
        InfluenceOraclePruner(
            scorer=DeterministicActionScorer(),
        ).prune(request)


def test_hf_scorer_is_lazy_and_forces_local_files() -> None:
    scorer = HFLogLikelihoodScorer("/models/not-loaded")
    assert scorer.is_loaded is False
    assert build_pruner({"model_path": "/models/not-loaded"}).scorer.is_loaded is False
    with pytest.raises(ValueError, match="local_files_only"):
        HFLogLikelihoodScorer(
            "/models/not-loaded",
            local_files_only=False,
        )
