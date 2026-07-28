from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from tf_pruning.budgets import LengthAwareBudget
from tf_pruning.io import read_jsonl, write_jsonl
from tf_pruning.protocol import Pruner, PruningRequest

from .metrics import ReplayLabels, aggregate_metrics, score_example


def load_replay_examples(
    path: str | Path,
) -> Iterable[tuple[PruningRequest, ReplayLabels, dict[str, Any]]]:
    for row_index, payload in enumerate(read_jsonl(path), start=1):
        request_payload = payload.get("request", payload)
        if not isinstance(request_payload, Mapping):
            raise ValueError(f"row {row_index}: request must be an object")
        request = PruningRequest.from_dict(request_payload)
        if request.request_id is None:
            request = replace(request, request_id=f"row-{row_index}")
        yield request, ReplayLabels.from_dict(payload), payload


def run_replay(
    pruner: Pruner,
    input_path: str | Path,
    output_dir: str | Path,
    *,
    budget_schedule: LengthAwareBudget | None = None,
    continue_on_error: bool = False,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for request, labels, _source in load_replay_examples(input_path):
        if budget_schedule is not None:
            request = replace(
                request,
                budget=budget_schedule.for_line_count(len(request.lines)),
            )
        try:
            result = pruner.prune(request)
        except Exception as exc:
            if not continue_on_error:
                raise
            errors.append(
                {
                    "request_id": request.request_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
        result_rows.append(result.to_dict())
        metric_rows.append(score_example(request, result, labels))

    summary = aggregate_metrics(metric_rows)
    summary.update(
        {
            "method": getattr(pruner, "name", type(pruner).__name__),
            "input": str(Path(input_path)),
            "successful_samples": len(metric_rows),
            "failed_samples": len(errors),
        }
    )
    write_jsonl(output_root / "results.jsonl", result_rows)
    write_jsonl(output_root / "per_sample_metrics.jsonl", metric_rows)
    write_jsonl(output_root / "errors.jsonl", errors)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary
