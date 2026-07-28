from __future__ import annotations

import math

import pytest

from tasks.hidden_state_similarity import (
    HiddenStateSimilarityPruner,
    build_pruner,
)
from tf_pruning.protocol import BudgetConfig, PruningRequest


def _budget(lines: int = 1) -> BudgetConfig:
    return BudgetConfig(
        keep_ratio=1.0,
        min_lines=lines,
        max_lines=lines,
        no_prune_below=0,
        context_window=0,
    )


def test_build_pruner_and_plain_list_input_use_shared_budget() -> None:
    request = PruningRequest(
        text="query match\nirrelevant\npartial match",
        budget=_budget(1),
        metadata={
            "hidden_states": [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]],
            "token_to_line": [1, 2, 3],
            "query_anchor": [1.0, 0.0],
        },
    )

    pruner = build_pruner({"anchor_weights": {"query": 1.0}})
    result = pruner.prune(request)

    assert isinstance(pruner, HiddenStateSimilarityPruner)
    assert result.kept_line_numbers == (1,)
    assert result.metadata["source"] == "metadata"
    assert result.metadata["anchors_used"] == ["query"]
    assert len(result.line_scores) == 3


@pytest.mark.parametrize("pooling", ["mean", "max", "last", "last-4"])
def test_supported_pooling_modes(pooling: str) -> None:
    # Four layers exercise last-4; each line has two tokens for token pooling.
    layers = [
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]],
    ]
    request = PruningRequest(
        text="first\nsecond",
        budget=_budget(1),
        metadata={
            "hidden_states": layers,
            "token_to_line": [1, 1, 2, 2],
            "anchors": {"query": [1.0, 0.0]},
        },
    )

    result = build_pruner({"pooling": pooling, "anchor_weights": {"query": 1.0}}).prune(request)

    assert result.kept_line_numbers == (1,)
    assert result.metadata["pooling"] == pooling
    assert result.metadata["layers_available"] == 4


def test_zero_based_token_line_map_and_all_anchor_sources_are_fused() -> None:
    request = PruningRequest(
        text="one\ntwo\nthree",
        budget=_budget(2),
        metadata={
            "hidden_states": [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.7, 0.7],
                [1.0, 0.0],
            ],
            "token_to_line": [0, 1, 2, -1],
            "anchors": {
                "query": [1.0, 0.0],
                "tool": [1.0, 0.0],
                "error": [0.0, 1.0],
                "decode": [1.0, 0.0],
            },
        },
    )
    result = build_pruner(
        {
            "line_map_base": "auto",
            "anchor_weights": {
                "query": 1.0,
                "tool": 1.0,
                "error": 1.0,
                "decode": 1.0,
            },
        }
    ).prune(request)

    assert result.kept_line_numbers == (1, 3)
    assert result.metadata["mapped_token_count"] == 3
    assert result.metadata["anchors_used"] == [
        "query",
        "tool",
        "error",
        "decode",
    ]
    assert all(
        any(reason.startswith("decode_cosine=") for reason in line.reasons)
        for line in result.line_scores
    )


def test_anchor_token_indices_and_automatic_error_anchor() -> None:
    request = PruningRequest(
        text="normal\nERROR: exploded\nnormal again",
        budget=_budget(1),
        metadata={
            "hidden_states": [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]],
            "token_to_line": [1, 2, 3],
            "anchor_token_indices": {"query": [0]},
        },
    )
    result = build_pruner(
        {
            "anchor_weights": {"query": 0.1, "error": 0.9},
            "derive_error_anchor": True,
        }
    ).prune(request)

    assert result.kept_line_numbers == (2,)
    assert result.metadata["anchors_used"] == ["query", "error"]


def test_npz_offline_adapter(tmp_path) -> None:
    np = pytest.importorskip("numpy")
    source = tmp_path / "states.npz"
    np.savez(
        source,
        hidden_states=np.asarray([[0.0, 1.0], [1.0, 0.0]]),
        token_to_line=np.asarray([1, 2]),
        query_anchor=np.asarray([1.0, 0.0]),
    )
    request = PruningRequest(
        text="low\nhigh",
        budget=_budget(1),
        metadata={"hidden_states_path": str(source)},
    )

    result = build_pruner({"anchor_weights": {"query": 1.0}}).prune(request)

    assert result.kept_line_numbers == (2,)
    assert result.metadata["source"] == "npz"
    assert math.isclose(result.line_scores[1].score, 1.0)


def test_rejects_mismatched_token_map() -> None:
    request = PruningRequest(
        text="one\ntwo",
        budget=_budget(1),
        metadata={
            "hidden_states": [[1.0, 0.0], [0.0, 1.0]],
            "token_to_line": [1],
            "query_anchor": [1.0, 0.0],
        },
    )
    with pytest.raises(ValueError, match="token_to_line length"):
        build_pruner().prune(request)
