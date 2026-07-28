from __future__ import annotations

import json

from evaluation.matrix import load_matrix, mark_recall_retention_pareto


def test_matrix_loader_and_pareto(tmp_path) -> None:
    for name, retention, recall in (
        ("keep_0.7", 0.7, 1.0),
        ("keep_0.5", 0.5, 0.9),
        ("keep_0.6", 0.6, 0.8),
    ):
        run = tmp_path / name
        run.mkdir()
        (run / "summary.json").write_text(
            json.dumps(
                {
                    "method": "fake",
                    "samples": 1,
                    "retention_ratio": retention,
                    "estimated_token_retention_ratio": retention,
                    "macro_line_recall": recall,
                    "macro_line_f1": recall,
                    "macro_required_line_recall": recall,
                    "critical_miss_rate": 0.0,
                    "latency_ms": {"mean": 1.0, "p95": 1.0},
                    "model_forward_count": 0,
                }
            ),
            encoding="utf-8",
        )
    rows = mark_recall_retention_pareto(load_matrix(tmp_path))
    by_budget = {row["budget"]: row for row in rows}
    assert by_budget["0.7"]["pareto"] is True
    assert by_budget["0.5"]["pareto"] is True
    assert by_budget["0.6"]["pareto"] is False
