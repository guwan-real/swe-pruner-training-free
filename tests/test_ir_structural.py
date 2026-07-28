from __future__ import annotations

import json

import pytest

from tasks.ir_structural import (
    IRStructuralPruner,
    build_pruner,
)
from tf_pruning.protocol import BudgetConfig, PruningRequest


def test_mixed_retrieval_structure_and_window_expansion() -> None:
    source = "\n".join(
        [
            "import json",
            "",
            "def unrelated():",
            "    value = 10",
            "    return value",
            "",
            "def load_config(path):",
            "    raw = path.read_text()",
            "    return json.loads(raw)",
            "",
            "config_loader.py:88: failed to load config",
            "unimportant footer",
        ]
    )
    request = PruningRequest(
        text=source,
        query="load_config failure",
        path="src/config_loader.py",
        recent_context=("json config error",),
        budget=BudgetConfig(
            keep_ratio=0.58,
            no_prune_below=0,
            context_window=1,
        ),
        request_id="ir-1",
    )

    result = build_pruner({"scoring_window": 0}).prune(request)

    assert result.method == "ir_structural"
    assert result.request_id == "ir-1"
    assert result.kept_line_count == 7
    assert {1, 3, 7}.issubset(result.kept_line_numbers)
    assert 9 in result.kept_line_numbers
    assert any(
        "window" in score.reasons and score.line_no in result.kept_line_numbers
        for score in result.line_scores
    )
    assert 11 in result.kept_line_numbers
    assert result.metadata["training_free"] is True
    assert result.metadata["structural_anchor_lines"] == [1, 3, 7]
    assert "... <" in result.pruned_text

    path_line = result.line_scores[10]
    assert path_line.line_no == 11
    assert "path" in path_line.reasons
    assert "recent" in path_line.reasons


def test_bm25_prioritises_rare_query_term_under_hard_budget() -> None:
    text = "\n".join(
        [
            "ordinary setup",
            "ordinary details",
            "ordinary cleanup",
            "rare_widget_factory builds the requested component",
            "ordinary footer",
        ]
    )
    request = PruningRequest(
        text=text,
        query="rare_widget_factory",
        budget=BudgetConfig(
            keep_ratio=0.2,
            min_lines=1,
            no_prune_below=0,
            context_window=0,
        ),
    )

    result = IRStructuralPruner().prune(request)

    assert result.kept_line_numbers == (4,)
    assert "bm25" in result.line_scores[3].reasons
    assert "identifier" in result.line_scores[3].reasons


def test_nested_weight_config_and_invalid_config() -> None:
    pruner = build_pruner(
        {
            "weights": {
                "bm25": 2.5,
                "identifier": 4.0,
            },
            "expansion_top_k": 2,
        }
    )
    assert pruner.config.bm25_weight == 2.5
    assert pruner.config.identifier_weight == 4.0

    with pytest.raises(ValueError, match="unknown IR structural"):
        build_pruner({"not_a_real_option": True})


def test_cli_writes_common_result_schema(tmp_path) -> None:
    from tasks.ir_structural.cli import main

    input_path = tmp_path / "requests.jsonl"
    output_path = tmp_path / "results.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "text": "alpha\nrare target\nomega",
                "query": "target",
                "budget": {
                    "keep_ratio": 0.34,
                    "no_prune_below": 0,
                    "context_window": 0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(["--input", str(input_path), "--output", str(output_path)]) == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["method"] == "ir_structural"
    assert payload["kept_line_numbers"] == [2]
