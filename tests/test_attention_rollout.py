from __future__ import annotations

import pytest

from tasks.attention_rollout import AttentionRolloutPruner, build_pruner
from tf_pruning.protocol import BudgetConfig, PruningRequest


def _budget(lines: int) -> BudgetConfig:
    return BudgetConfig(
        keep_ratio=1.0,
        min_lines=lines,
        max_lines=lines,
        no_prune_below=0,
        context_window=0,
    )


def test_attention_mass_aggregates_layers_heads_and_decode_steps() -> None:
    # Shape: [layers=2, heads=2, steps=3, tokens=3].
    attention = [
        [
            [[0.1, 0.8, 0.1], [0.1, 0.7, 0.2], [0.1, 0.9, 0.0]],
            [[0.2, 0.7, 0.1], [0.1, 0.8, 0.1], [0.0, 0.9, 0.1]],
        ],
        [
            [[0.1, 0.9, 0.0], [0.1, 0.8, 0.1], [0.2, 0.7, 0.1]],
            [[0.0, 0.9, 0.1], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1]],
        ],
    ]
    request = PruningRequest(
        text="first\nimportant\nthird",
        budget=_budget(1),
        metadata={"attention": attention, "token_to_line": [1, 2, 3]},
    )

    pruner = build_pruner(
        {
            "layers": "all",
            "heads": "all",
            "decode_steps": "all",
            "local_seed_count": 0,
            "structure_floor": 0.0,
        }
    )
    result = pruner.prune(request)

    assert isinstance(pruner, AttentionRolloutPruner)
    assert result.kept_line_numbers == (2,)
    assert result.metadata["selected_axes"] == {
        "layers": [0, 1],
        "heads": [0, 1],
        "decode_steps": [0, 1, 2],
    }


def test_attention_sink_is_masked_before_line_pooling() -> None:
    request = PruningRequest(
        text="sink line\nreal signal\nweak",
        budget=_budget(1),
        metadata={
            "attention": [100.0, 0.8, 0.2],
            "token_to_line": [1, 2, 3],
        },
    )
    result = build_pruner(
        {
            "layers": "all",
            "decode_steps": "all",
            "sink_first_tokens": 1,
            "local_seed_count": 0,
            "structure_floor": 0.0,
        }
    ).prune(request)

    assert result.kept_line_numbers == (2,)
    assert result.metadata["sink_token_indices"] == [0]
    assert result.line_scores[0].score == 0.0


def test_structure_and_local_floors_change_scores_without_breaking_budget() -> None:
    request = PruningRequest(
        text="def target():\n    context\nunrelated\nstrong",
        budget=_budget(2),
        metadata={
            "attention": [0.0, 0.0, 0.0, 1.0],
            "token_to_line": [1, 2, 3, 4],
        },
    )
    result = build_pruner(
        {
            "layers": "all",
            "decode_steps": "all",
            "structure_floor": 0.7,
            "local_floor": 0.5,
            "local_window": 1,
            "local_seed_count": 1,
            "selection_mode": "hard_budget",
        }
    ).prune(request)

    assert result.kept_line_numbers == (1, 4)
    assert len(result.kept_line_numbers) == 2
    assert result.line_scores[0].score == 0.7
    assert result.line_scores[2].score == 0.5
    assert "structure_floor=0.700000" in result.line_scores[0].reasons
    assert "local_floor_from=4" in result.line_scores[2].reasons


def test_top_p_mode_can_return_fewer_than_hard_budget() -> None:
    request = PruningRequest(
        text="dominant\nsmall\nsmallest",
        budget=_budget(3),
        metadata={
            "attention": [0.95, 0.04, 0.01],
            "token_to_line": [1, 2, 3],
        },
    )
    result = build_pruner(
        {
            "layers": "all",
            "decode_steps": "all",
            "top_p": 0.9,
            "selection_mode": "top_p",
            "local_seed_count": 0,
            "structure_floor": 0.0,
        }
    ).prune(request)

    # no_prune_below is zero, but a three-line hard cap does not force Top-P
    # to fill the cap.
    assert result.kept_line_numbers == (1,)
    assert result.metadata["top_p_line_numbers"] == [1]


def test_rollout_composes_square_attention_and_uses_decode_rows() -> None:
    # One layer/head. Decode row 2 attends mostly to response token on line 2.
    attention = [
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.1, 0.8, 0.1],
            ]
        ]
    ]
    request = PruningRequest(
        text="line one\nline two",
        budget=_budget(1),
        metadata={
            "attention": attention,
            "token_to_line": [1, 2, -1],
            "decode_token_indices": [2],
        },
    )
    result = build_pruner(
        {
            "method": "rollout",
            "layers": "all",
            "heads": "all",
            "rollout_residual_weight": 0.0,
            "local_seed_count": 0,
            "structure_floor": 0.0,
        }
    ).prune(request)

    assert result.kept_line_numbers == (2,)
    assert result.metadata["attention_layout"] == [
        "layers",
        "heads",
        "queries",
        "keys",
    ]
    assert result.metadata["selected_axes"]["decode_steps"] == [2]


def test_npz_attention_input(tmp_path) -> None:
    np = pytest.importorskip("numpy")
    source = tmp_path / "attention.npz"
    np.savez(
        source,
        attention=np.asarray([[0.2, 0.8]]),
        token_to_line=np.asarray([1, 2]),
    )
    request = PruningRequest(
        text="low\nhigh",
        budget=_budget(1),
        metadata={"attention_path": str(source)},
    )
    result = build_pruner(
        {
            "layers": "all",
            "decode_steps": "all",
            "local_seed_count": 0,
            "structure_floor": 0.0,
        }
    ).prune(request)

    assert result.kept_line_numbers == (2,)
    assert result.metadata["source"] == "npz"
