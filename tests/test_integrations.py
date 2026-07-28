from __future__ import annotations

import pytest

from integrations.http_server import (
    CompatPruningService,
    threshold_to_keep_ratio,
)
from integrations.middleware import TrainingFreeMiddleware, infer_tool_type
from tasks.ir_structural import build_pruner
from tf_pruning.budgets import LengthAwareBudget, LengthBand


class BrokenPruner:
    name = "broken"

    def prune(self, request):
        raise RuntimeError("expected failure")


def _schedule() -> LengthAwareBudget:
    return LengthAwareBudget(
        bands=(LengthBand(max_lines=None, keep_ratio=0.5),),
        no_prune_below=0,
        context_window=0,
    )


def test_middleware_prunes_and_infers_tool_type() -> None:
    middleware = TrainingFreeMiddleware(
        build_pruner({"preserve_structure": False}),
        budget_schedule=_schedule(),
    )
    outcome = middleware.prune_tool_response(
        "alpha\nneedle = 1\nomega\nunused",
        query="needle",
        tool_name="read_file",
        path="module.py",
    )
    assert outcome.pruned is True
    assert outcome.result is not None
    assert outcome.result.kept_line_count == 2
    assert infer_tool_type(command="git diff -- src/x.py") == "diff"
    assert infer_tool_type(command="pytest -q") == "test_log"
    assert infer_tool_type(command="rg needle src") == "grep"


def test_middleware_is_fail_open_by_default() -> None:
    outcome = TrainingFreeMiddleware(BrokenPruner()).prune_tool_response("original")
    assert outcome.text == "original"
    assert outcome.pruned is False
    assert "expected failure" in str(outcome.error)

    with pytest.raises(RuntimeError, match="expected failure"):
        TrainingFreeMiddleware(
            BrokenPruner(),
            fail_open=False,
        ).prune_tool_response("original")


def test_official_shape_response_and_threshold_mapping() -> None:
    service = CompatPruningService(
        build_pruner({"preserve_structure": False}),
        default_no_prune_below=0,
    )
    response = service.prune_payload(
        {
            "query": "needle",
            "code": "alpha\nneedle = 1\nomega\nunused",
            "threshold": 0.5,
        }
    )
    assert response["method"] == "ir_structural"
    assert response["error_msg"] is None
    assert len(response["kept_frags"]) == 2
    assert response["score_semantics"] == "method_native_max_kept_line_score"
    assert response["token_scores_granularity"] == "line"
    assert set(
        (
            "score",
            "pruned_code",
            "token_scores",
            "kept_frags",
            "origin_token_cnt",
            "left_token_cnt",
            "model_input_token_cnt",
        )
    ).issubset(response)
    assert threshold_to_keep_ratio(0.8) == pytest.approx(0.2)


def test_official_shape_service_fails_open_for_missing_model_signal() -> None:
    from tasks.hidden_state_similarity import build_pruner as build_hidden

    response = CompatPruningService(build_hidden()).prune_payload(
        {
            "query": "needle",
            "code": "alpha\nneedle",
            "no_prune_below": 0,
        }
    )
    assert response["fail_open"] is True
    assert response["pruned_code"] == "alpha\nneedle"
    assert response["score_semantics"] == "fail_open_constant"
    assert "hidden_states" in response["error_msg"]
