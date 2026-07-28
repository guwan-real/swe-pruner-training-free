from __future__ import annotations

import json
from pathlib import Path

import pytest

from tf_pruning.budgets import LengthAwareBudget
from tf_pruning.protocol import BudgetConfig, PruningRequest, PruningResult
from tf_pruning.registry import available_methods, build_pruner
from tf_pruning.selection import render_pruned_text, select_line_numbers


def test_budget_validation_and_target() -> None:
    with pytest.raises(ValueError):
        BudgetConfig(keep_ratio=0)
    budget = BudgetConfig(
        keep_ratio=0.5,
        min_lines=2,
        no_prune_below=0,
    )
    assert budget.target_lines(5) == 2


def test_selection_respects_budget_and_mandatory_lines() -> None:
    budget = BudgetConfig(
        keep_ratio=0.4,
        min_lines=1,
        no_prune_below=0,
        context_window=1,
    )
    selected = select_line_numbers(
        [0.0, 2.0, 0.1, 1.0, 0.2],
        budget,
        mandatory=(4,),
        expansion_seeds=(2,),
    )
    assert selected == (2, 4)


def test_renderer_makes_omissions_explicit() -> None:
    rendered = render_pruned_text(["a", "b", "c", "d"], (2, 4))
    assert "<1 lines pruned>" in rendered
    assert "     2 | b" in rendered
    assert "     4 | d" in rendered


def test_length_aware_budget_selects_band() -> None:
    schedule = LengthAwareBudget.from_dict(
        {
            "bands": [
                {"max_lines": 20, "keep_ratio": 1.0},
                {"max_lines": None, "keep_ratio": 0.25},
            ],
            "no_prune_below": 0,
        }
    )
    assert schedule.for_line_count(10).keep_ratio == 1.0
    assert schedule.for_line_count(100).keep_ratio == 0.25


def test_protocol_round_trip_shape() -> None:
    request = PruningRequest.from_dict(
        {
            "request_id": "r1",
            "text": "a\nb",
            "budget": {"keep_ratio": 1.0},
        }
    )
    result = PruningResult(
        method="test",
        request_id=request.request_id,
        original_line_count=2,
        kept_line_numbers=(1,),
        pruned_text="a",
    )
    payload = result.to_dict()
    assert payload["request_id"] == "r1"
    assert payload["retention_ratio"] == 0.5


def test_all_method_factories_accept_their_example_config() -> None:
    root = Path(__file__).parents[1]
    for method in available_methods():
        payload = json.loads(
            (root / "tasks" / method / "config.example.json").read_text(encoding="utf-8")
        )
        pruner = build_pruner(method, payload)
        assert pruner.name == method
