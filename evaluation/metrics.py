from __future__ import annotations

import math
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from tf_pruning.protocol import PruningRequest, PruningResult


@dataclass(frozen=True)
class ReplayLabels:
    gold_line_numbers: frozenset[int] = frozenset()
    required_line_numbers: frozenset[int] = frozenset()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReplayLabels":
        gold = payload.get(
            "gold_line_numbers",
            payload.get("important_line_numbers", ()),
        )
        required = payload.get(
            "required_line_numbers",
            payload.get("answer_line_numbers", ()),
        )
        return cls(
            gold_line_numbers=frozenset(int(item) for item in gold),
            required_line_numbers=frozenset(int(item) for item in required),
        )


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def score_example(
    request: PruningRequest,
    result: PruningResult,
    labels: ReplayLabels,
) -> dict[str, Any]:
    kept = set(result.kept_line_numbers)
    gold = set(labels.gold_line_numbers)
    required = set(labels.required_line_numbers)
    true_positive = len(kept & gold)
    precision = safe_divide(true_positive, len(kept)) if gold else None
    recall = safe_divide(true_positive, len(gold)) if gold else None
    f1 = (
        safe_divide(2.0 * precision * recall, precision + recall)
        if precision is not None and recall is not None
        else None
    )
    required_recall = safe_divide(len(kept & required), len(required)) if required else None
    original_tokens = len(re.findall(r"\w+|[^\w\s]", request.text))
    kept_source = "\n".join(
        request.lines[line_no - 1]
        for line_no in result.kept_line_numbers
        if 1 <= line_no <= len(request.lines)
    )
    kept_tokens = len(re.findall(r"\w+|[^\w\s]", kept_source))
    return {
        "request_id": result.request_id,
        "method": result.method,
        "tool_type": request.tool_type,
        "original_line_count": result.original_line_count,
        "kept_line_count": result.kept_line_count,
        "retention_ratio": result.retention_ratio,
        "line_savings_ratio": 1.0 - result.retention_ratio,
        "estimated_original_tokens": original_tokens,
        "estimated_kept_source_tokens": kept_tokens,
        "estimated_token_retention_ratio": safe_divide(kept_tokens, original_tokens)
        if original_tokens
        else 1.0,
        "line_precision": precision,
        "line_recall": recall,
        "line_f1": f1,
        "required_line_recall": required_recall,
        "critical_miss": (None if required_recall is None else required_recall < 1.0),
        "latency_ms": result.latency_ms,
        "model_forward_count": int(
            result.metadata.get(
                "model_forward_count",
                result.metadata.get(
                    "scorer_calls",
                    result.metadata.get("evaluations", 0),
                ),
            )
        ),
    }


def _mean_defined(rows: Iterable[Mapping[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return statistics.fmean(values) if values else None


def aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    original_lines = sum(int(row["original_line_count"]) for row in rows)
    kept_lines = sum(int(row["kept_line_count"]) for row in rows)
    original_tokens = sum(int(row["estimated_original_tokens"]) for row in rows)
    kept_tokens = sum(int(row["estimated_kept_source_tokens"]) for row in rows)
    labeled_required = [row for row in rows if row.get("critical_miss") is not None]
    latencies = [float(row["latency_ms"]) for row in rows]
    summary: dict[str, Any] = {
        "samples": len(rows),
        "original_lines": original_lines,
        "kept_lines": kept_lines,
        "retention_ratio": safe_divide(kept_lines, original_lines) if original_lines else 1.0,
        "line_savings_ratio": 1.0
        - (safe_divide(kept_lines, original_lines) if original_lines else 1.0),
        "estimated_original_tokens": original_tokens,
        "estimated_kept_source_tokens": kept_tokens,
        "estimated_token_retention_ratio": (
            safe_divide(kept_tokens, original_tokens) if original_tokens else 1.0
        ),
        "model_forward_count": sum(int(row.get("model_forward_count", 0)) for row in rows),
        "macro_line_precision": _mean_defined(rows, "line_precision"),
        "macro_line_recall": _mean_defined(rows, "line_recall"),
        "macro_line_f1": _mean_defined(rows, "line_f1"),
        "macro_required_line_recall": _mean_defined(rows, "required_line_recall"),
        "critical_miss_rate": (
            safe_divide(
                sum(bool(row["critical_miss"]) for row in labeled_required),
                len(labeled_required),
            )
            if labeled_required
            else None
        ),
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
        },
    }

    by_tool: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_tool[str(row.get("tool_type", "unknown"))].append(row)
    summary["by_tool_type"] = (
        {
            tool_type: {
                key: value
                for key, value in aggregate_metrics(tool_rows).items()
                if key != "by_tool_type"
            }
            for tool_type, tool_rows in sorted(by_tool.items())
        }
        if len(by_tool) > 1
        else {}
    )
    return summary
