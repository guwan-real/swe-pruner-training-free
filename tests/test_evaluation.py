from __future__ import annotations

from evaluation.metrics import ReplayLabels, aggregate_metrics, score_example
from tf_pruning.protocol import PruningRequest, PruningResult


def test_metrics_measure_recall_retention_and_critical_miss() -> None:
    request = PruningRequest(text="a\nb\nc\nd", tool_type="source")
    result = PruningResult(
        method="fake",
        original_line_count=4,
        kept_line_numbers=(2, 4),
        pruned_text="b\nd",
        latency_ms=2.0,
    )
    row = score_example(
        request,
        result,
        ReplayLabels(
            gold_line_numbers=frozenset({2, 3}),
            required_line_numbers=frozenset({2, 3}),
        ),
    )
    assert row["line_precision"] == 0.5
    assert row["line_recall"] == 0.5
    assert row["retention_ratio"] == 0.5
    assert row["critical_miss"] is True
    assert row["estimated_token_retention_ratio"] == 0.5
    summary = aggregate_metrics([row])
    assert summary["samples"] == 1
    assert summary["critical_miss_rate"] == 1.0
