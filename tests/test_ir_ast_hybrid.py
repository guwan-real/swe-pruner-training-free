from __future__ import annotations

from tasks.ir_ast_hybrid.pruner import (
    IRASTHybridConfig,
    IRASTHybridPruner,
    _rank_percentiles,
)
from tf_pruning.protocol import BudgetConfig, PruningRequest


def test_rank_percentiles_are_deterministic_for_ties() -> None:
    assert _rank_percentiles([2.0, 2.0, 1.0]) == [1.0, 1.0, 0.0]


def test_hybrid_keeps_critical_execution_evidence_with_hard_budget() -> None:
    text = "\n".join(
        [
            "import os",
            "",
            "def unrelated():",
            "    return 1",
            "",
            "def load_timeout():",
            "    value = os.environ['TIMEOUT']",
            "    raise RuntimeError('timeout invalid')",
            "    return value",
            "",
            "print(load_timeout())",
            "footer",
        ]
    )
    request = PruningRequest(
        text=text,
        query="Where is timeout validation and RuntimeError handled?",
        path="settings.py",
        budget=BudgetConfig(
            keep_ratio=0.5,
            no_prune_below=0,
            context_window=1,
        ),
    )
    result = IRASTHybridPruner().prune(request)

    assert result.method == "ir_ast_hybrid"
    assert result.kept_line_count == 6
    assert 6 in result.kept_line_numbers
    assert 8 in result.kept_line_numbers
    assert result.metadata["model_forward_count"] == 0
    assert result.metadata["training_free"] is True


def test_hybrid_nested_config_is_validated() -> None:
    config = IRASTHybridConfig.from_mapping(
        {
            "weights": {"ir": 0.7, "execution_ast": 0.3},
            "ir": {"show_line_numbers": False},
            "execution_ast": {"show_line_numbers": False},
        }
    )
    assert config.ir_weight == 0.7
    assert config.execution_ast_weight == 0.3
    assert config.ir.show_line_numbers is False
