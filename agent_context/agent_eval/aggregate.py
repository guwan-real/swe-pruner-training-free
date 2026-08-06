from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _usage(message: dict[str, Any]) -> tuple[int, int, int, int | None]:
    extra = message.get("extra")
    response = extra.get("response", {}) if isinstance(extra, dict) else {}
    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    if not isinstance(usage, dict):
        return 0, 0, 0, None
    prompt = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    completion = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
    total = int(usage.get("total_tokens", 0) or prompt + completion)
    details = usage.get("prompt_tokens_details", usage.get("input_tokens_details"))
    cached: int | None = None
    if isinstance(details, dict) and details.get("cached_tokens") is not None:
        cached = int(details["cached_tokens"] or 0)
    elif usage.get("cached_prompt_tokens") is not None:
        cached = int(usage["cached_prompt_tokens"] or 0)
    return prompt, completion, total, cached


def _wall_time(path: Path) -> float | None:
    started = path / "started_at"
    ended = path / "ended_at"
    if not started.is_file() or not ended.is_file():
        return None
    start = datetime.fromisoformat(
        started.read_text(encoding="utf-8").strip().replace("Z", "+00:00")
    )
    end = datetime.fromisoformat(ended.read_text(encoding="utf-8").strip().replace("Z", "+00:00"))
    return max(0.0, (end - start).total_seconds())


def _read_prometheus(path: Path) -> Counter[str]:
    totals: Counter[str] = Counter()
    if not path.is_file():
        return totals
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        name = fields[0].split("{", 1)[0]
        try:
            value = float(fields[1])
        except ValueError:
            continue
        if math.isfinite(value):
            totals[name] += value
    return totals


def _metric_delta(
    before: Counter[str],
    after: Counter[str],
    *aliases: str,
) -> float | None:
    for name in aliases:
        if name not in before or name not in after:
            continue
        delta = after[name] - before[name]
        return delta if delta >= 0 else None
    return None


def _mean(total: float | None, count: float | None) -> float | None:
    return total / count if total is not None and count not in (None, 0) else None


def _vllm_metrics(path: Path) -> dict[str, Any]:
    before = _read_prometheus(path / "vllm_metrics_before.prom")
    after = _read_prometheus(path / "vllm_metrics_after.prom")
    available = bool(before and after)
    prompt_tokens = _metric_delta(
        before,
        after,
        "vllm:prompt_tokens_total",
        "vllm_prompt_tokens_total",
    )
    prefix_queries = _metric_delta(
        before,
        after,
        "vllm:prefix_cache_queries_total",
        "vllm_prefix_cache_queries_total",
    )
    prefix_hits = _metric_delta(
        before,
        after,
        "vllm:prefix_cache_hits_total",
        "vllm_prefix_cache_hits_total",
    )
    requests = _metric_delta(
        before,
        after,
        "vllm:request_success_total",
        "vllm_request_success_total",
    )
    ttft_total = _metric_delta(
        before,
        after,
        "vllm:time_to_first_token_seconds_sum",
        "vllm_time_to_first_token_seconds_sum",
    )
    ttft_count = _metric_delta(
        before,
        after,
        "vllm:time_to_first_token_seconds_count",
        "vllm_time_to_first_token_seconds_count",
    )
    prefill_total = _metric_delta(
        before,
        after,
        "vllm:request_prefill_time_seconds_sum",
        "vllm_request_prefill_time_seconds_sum",
    )
    prefill_count = _metric_delta(
        before,
        after,
        "vllm:request_prefill_time_seconds_count",
        "vllm_request_prefill_time_seconds_count",
    )
    e2e_total = _metric_delta(
        before,
        after,
        "vllm:e2e_request_latency_seconds_sum",
        "vllm_e2e_request_latency_seconds_sum",
    )
    e2e_count = _metric_delta(
        before,
        after,
        "vllm:e2e_request_latency_seconds_count",
        "vllm_e2e_request_latency_seconds_count",
    )
    return {
        "vllm_metrics_available": available,
        "vllm_prompt_tokens": prompt_tokens,
        "vllm_prefix_cache_queries": prefix_queries,
        "vllm_prefix_cache_hits": prefix_hits,
        "vllm_prefix_cache_hit_ratio": (
            prefix_hits / prefix_queries
            if prefix_hits is not None and prefix_queries not in (None, 0)
            else None
        ),
        "vllm_successful_requests": requests,
        "vllm_ttft_seconds_total": ttft_total,
        "vllm_ttft_count": ttft_count,
        "vllm_ttft_seconds_mean": _mean(ttft_total, ttft_count),
        "vllm_prefill_seconds_total": prefill_total,
        "vllm_prefill_count": prefill_count,
        "vllm_prefill_seconds_mean": _mean(prefill_total, prefill_count),
        "vllm_e2e_latency_seconds_total": e2e_total,
        "vllm_e2e_latency_count": e2e_count,
        "vllm_e2e_latency_seconds_mean": _mean(e2e_total, e2e_count),
    }


def _count_value(payload: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, list):
            return len(value)
    return None


def _prediction_ids(predictions: Any) -> list[str]:
    if isinstance(predictions, dict):
        values = predictions.items()
    elif isinstance(predictions, list):
        values = enumerate(predictions)
    else:
        return []
    ids: list[str] = []
    for key, value in values:
        if isinstance(value, dict):
            value_id = value.get("instance_id") or value.get("task_id") or key
        else:
            value_id = key
        ids.append(str(value_id))
    return sorted(ids)


def _grader_metrics(run_root: Path, arm: str) -> tuple[int, int] | None:
    root = run_root / "grade" / arm
    if not root.is_dir():
        return None
    for candidate in sorted(
        root.rglob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True
    ):
        if candidate.name in {"preds.json", "preds.jsonl"}:
            continue
        try:
            payload = _read_json(candidate)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        resolved = _count_value(payload, "resolved_instances", "resolved_ids", "resolved")
        graded = _count_value(
            payload,
            "completed_instances",
            "completed_ids",
            "submitted_instances",
            "submitted_ids",
            "total_instances",
        )
        if resolved is not None and graded is not None:
            return resolved, graded
    return None


def _add_manifest(
    manifest: dict[str, Any],
    totals: Counter[str],
    levels: Counter[str],
    kinds: Counter[str],
    kind_levels: Counter[str],
) -> None:
    totals["prompt_manifests"] += 1
    full = int(manifest.get("full_observation_tokens", 0) or 0)
    selected = int(manifest.get("selected_observation_tokens", 0) or 0)
    totals["full_history_tokens"] += full
    totals["selected_history_tokens"] += selected
    totals["estimated_saved_tokens"] += int(
        manifest.get("estimated_tokens_saved", max(0, full - selected)) or 0
    )
    overflow = int(manifest.get("budget_overflow_tokens", 0) or 0)
    totals["budget_overflow_tokens"] += overflow
    totals["budget_overflow_prompts"] += int(overflow > 0)
    totals["context_view_switches"] += int(manifest.get("context_view_switches", 0) or 0)
    entries = manifest.get("entries", [])
    if not isinstance(entries, list):
        return
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        level = str(entry.get("selected_level", "unknown"))
        kind = str(entry.get("kind", "unknown"))
        levels[level] += 1
        kinds[kind] += 1
        kind_levels[f"{kind}:{level}"] += 1
        totals["compact_prompt_entries"] += int(level != "full")
        totals["selection_changed_entries"] += int(bool(entry.get("selection_changed")))


def summarize_arm(path: Path) -> dict[str, Any]:
    trajectories = sorted(path.rglob("*.traj.json"))
    trajectory_keys = [str(item.relative_to(path)) for item in trajectories]
    trajectory_ids = sorted(
        {
            (
                str(item.relative_to(path).parts[0])
                if len(item.relative_to(path).parts) > 1
                else item.name.removesuffix(".traj.json")
            )
            for item in trajectories
        }
    )
    predictions: Any = _read_json(path / "preds.json") if (path / "preds.json").is_file() else {}
    predictions = predictions if isinstance(predictions, dict) else {}
    calls = prompt_tokens = completion_tokens = total_tokens = 0
    cached_prompt_tokens = cache_detail_calls = usage_calls_total = 0
    tracked = untracked = 0
    totals: Counter[str] = Counter()
    selected_levels: Counter[str] = Counter()
    observation_kinds: Counter[str] = Counter()
    kind_levels: Counter[str] = Counter()
    tracking_reasons: Counter[str] = Counter()
    exit_statuses: Counter[str] = Counter()
    reports: list[dict[str, Any]] = []
    calls_per_trajectory: list[int] = []
    usage_complete = True
    for trajectory in trajectories:
        payload = _read_json(trajectory)
        if not isinstance(payload, dict):
            continue
        info = payload.get("info")
        reported_calls = 0
        if isinstance(info, dict):
            exit_statuses[str(info.get("exit_status", "unknown"))] += 1
            model_stats = info.get("model_stats", {})
            if isinstance(model_stats, dict):
                reported_calls = int(model_stats.get("api_calls", 0) or 0)
        messages = payload.get("messages", [])
        if not isinstance(messages, list):
            continue
        usage_calls = 0
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") == "assistant":
                prompt, completion, total, cached = _usage(message)
                if prompt or completion or total:
                    usage_calls += 1
                    usage_calls_total += 1
                    prompt_tokens += prompt
                    completion_tokens += completion
                    total_tokens += total
                    if cached is not None:
                        cache_detail_calls += 1
                        cached_prompt_tokens += cached
                manifest = message.get("agent_context_manifest")
                if isinstance(manifest, dict):
                    _add_manifest(
                        manifest,
                        totals,
                        selected_levels,
                        observation_kinds,
                        kind_levels,
                    )
            stats = message.get("agent_context_stats")
            if isinstance(stats, dict):
                status = str(stats.get("status", "unknown"))
                tracking_reasons[str(stats.get("reason", status))] += 1
                if status == "tracked":
                    tracked += 1
                elif status == "untracked":
                    untracked += 1
            report = message.get("agent_context_report")
            if isinstance(report, dict):
                reports.append(report)
                totals["committed_observations"] += int(
                    report.get("committed_observations", 0) or 0
                )
        trajectory_calls = max(reported_calls, usage_calls)
        usage_complete = usage_complete and usage_calls > 0 and usage_calls == trajectory_calls
        calls_per_trajectory.append(trajectory_calls)
        calls += trajectory_calls
    exit_code = None
    if (path / "exit_code").is_file():
        exit_code = int((path / "exit_code").read_text(encoding="utf-8").strip())
    cache_complete = (
        usage_calls_total > 0 and usage_complete and cache_detail_calls == usage_calls_total
    )
    usage_telemetry_complete = (
        usage_calls_total > 0 and usage_complete and usage_calls_total == calls
    )
    uncached_prompt_tokens = prompt_tokens - cached_prompt_tokens if cache_complete else None
    full_history_tokens = totals["full_history_tokens"]
    selected_history_tokens = totals["selected_history_tokens"]
    vllm = _vllm_metrics(path)
    vllm_metrics_isolated = bool(calls) and vllm["vllm_metrics_available"] and all(
        vllm[field] == calls
        for field in (
            "vllm_successful_requests",
            "vllm_ttft_count",
            "vllm_prefill_count",
            "vllm_e2e_latency_count",
        )
    )
    return {
        "arm": path.name,
        "predictions": len(predictions),
        "prediction_ids": _prediction_ids(predictions),
        "trajectories": len(trajectories),
        "trajectory_ids": trajectory_ids,
        "trajectory_keys": trajectory_keys,
        "submitted": exit_statuses.get("Submitted", 0),
        "all_tasks_submitted": bool(trajectories)
        and exit_statuses.get("Submitted", 0) == len(trajectories),
        "agent_api_calls": calls,
        "agent_api_calls_mean_per_task": (
            calls / len(calls_per_trajectory) if calls_per_trajectory else None
        ),
        "agent_api_calls_max_per_task": max(calls_per_trajectory, default=0),
        "agent_step_limit_hits": sum(value >= 100 for value in calls_per_trajectory),
        "agent_step_limit_exits": exit_statuses.get("LimitsExceeded", 0),
        "agent_prompt_tokens": prompt_tokens,
        "agent_cached_prompt_tokens": cached_prompt_tokens if cache_complete else None,
        "agent_uncached_prompt_tokens": uncached_prompt_tokens,
        "prompt_cache_hit_ratio": (
            cached_prompt_tokens / prompt_tokens if cache_complete and prompt_tokens else None
        ),
        "cache_telemetry_calls": cache_detail_calls,
        "usage_telemetry_complete": usage_telemetry_complete,
        "cache_telemetry_complete": cache_complete,
        "agent_completion_tokens": completion_tokens,
        "agent_total_tokens": total_tokens,
        "context_observations_tracked": tracked,
        "context_observations_untracked": untracked,
        "context_prompt_manifests": totals["prompt_manifests"],
        "context_full_history_tokens": full_history_tokens,
        "context_selected_history_tokens": selected_history_tokens,
        "context_estimated_tokens_saved": totals["estimated_saved_tokens"],
        "context_history_retention_ratio": (
            selected_history_tokens / full_history_tokens if full_history_tokens else None
        ),
        "context_compact_prompt_entries": totals["compact_prompt_entries"],
        "context_committed_observations": totals["committed_observations"],
        "context_view_switches": totals["context_view_switches"],
        "context_selection_changed_entries": totals["selection_changed_entries"],
        "context_budget_overflow_tokens": totals["budget_overflow_tokens"],
        "context_budget_overflow_prompts": totals["budget_overflow_prompts"],
        "runner_exit_code": exit_code,
        "runner_completed": exit_code is not None and (path / "ended_at").is_file(),
        "wall_time_seconds": _wall_time(path),
        "selected_levels": dict(sorted(selected_levels.items())),
        "observation_kinds": dict(sorted(observation_kinds.items())),
        "kind_levels": dict(sorted(kind_levels.items())),
        "tracking_reasons": dict(sorted(tracking_reasons.items())),
        "exit_statuses": dict(sorted(exit_statuses.items())),
        "trajectory_reports": reports,
        "resolved": None,
        "graded": None,
        "resolve_rate": None,
        "vllm_metrics_isolated": vllm_metrics_isolated,
        **vllm,
    }


def _ratio(value: int | float | None, reference: int | float | None) -> float | None:
    return value / reference if value is not None and reference not in (None, 0) else None


def _delta(value: int | float | None, reference: int | float | None) -> int | float | None:
    return value - reference if value is not None and reference is not None else None


def _add_reference_comparisons(rows: list[dict[str, Any]], reference_arm: str) -> None:
    reference = next((row for row in rows if row["arm"] == reference_arm), None)
    fields = {
        "agent_api_calls_delta_vs_reference": ("agent_api_calls", _delta, "base"),
        "agent_prompt_token_ratio_vs_reference": (
            "agent_prompt_tokens",
            _ratio,
            "usage",
        ),
        "agent_uncached_prompt_token_ratio_vs_reference": (
            "agent_uncached_prompt_tokens",
            _ratio,
            "cache",
        ),
        "wall_time_ratio_vs_reference": ("wall_time_seconds", _ratio, "base"),
        "vllm_prefix_cache_hit_ratio_delta_vs_reference": (
            "vllm_prefix_cache_hit_ratio",
            _delta,
            "vllm",
        ),
        "vllm_prefill_time_ratio_vs_reference": (
            "vllm_prefill_seconds_total",
            _ratio,
            "vllm",
        ),
        "vllm_ttft_ratio_vs_reference": (
            "vllm_ttft_seconds_total",
            _ratio,
            "vllm",
        ),
        "vllm_e2e_latency_ratio_vs_reference": (
            "vllm_e2e_latency_seconds_total",
            _ratio,
            "vllm",
        ),
        "resolve_rate_delta_vs_reference": ("resolve_rate", _delta, "grade"),
    }
    for row in rows:
        base_comparable = (
            reference is not None
            and reference["trajectories"] > 0
            and reference["runner_exit_code"] == 0
            and reference["runner_completed"]
            and reference["predictions"] == reference["trajectories"]
            and reference["all_tasks_submitted"]
            and row["runner_exit_code"] == 0
            and row["runner_completed"]
            and row["predictions"] == row["trajectories"]
            and row["all_tasks_submitted"]
            and row["prediction_ids"] == row["trajectory_ids"]
            and reference["prediction_ids"] == reference["trajectory_ids"]
            and row["prediction_ids"] == reference["prediction_ids"]
            and row["trajectory_keys"] == reference["trajectory_keys"]
        )
        comparable = {
            "base": base_comparable,
            "usage": (
                base_comparable
                and reference["usage_telemetry_complete"]
                and row["usage_telemetry_complete"]
            ),
            "cache": (
                base_comparable
                and reference["cache_telemetry_complete"]
                and row["cache_telemetry_complete"]
            ),
            "vllm": (
                base_comparable
                and reference["vllm_metrics_available"]
                and row["vllm_metrics_available"]
                and reference["vllm_metrics_isolated"]
                and row["vllm_metrics_isolated"]
            ),
            "grade": (
                base_comparable
                and reference["graded"] == reference["trajectories"]
                and row["graded"] == row["trajectories"]
            ),
        }
        for destination, (source, operation, requirement) in fields.items():
            row[destination] = (
                operation(row.get(source), reference.get(source))
                if comparable[requirement]
                else None
            )


def summarize_run(run_root: Path, *, reference_arm: str = "R") -> list[dict[str, Any]]:
    arms = run_root / "arms"
    if not arms.is_dir():
        raise FileNotFoundError(f"arms directory does not exist: {arms}")
    rows = [summarize_arm(path) for path in sorted(arms.iterdir()) if path.is_dir()]
    for row in rows:
        graded = _grader_metrics(run_root, row["arm"])
        if graded:
            row["resolved"], row["graded"] = graded
            row["resolve_rate"] = row["resolved"] / row["graded"] if row["graded"] else None
    _add_reference_comparisons(rows, reference_arm)
    return rows


FIELDS = [
    "arm",
    "predictions",
    "prediction_ids",
    "trajectories",
    "trajectory_ids",
    "submitted",
    "all_tasks_submitted",
    "agent_api_calls",
    "agent_api_calls_mean_per_task",
    "agent_api_calls_max_per_task",
    "agent_step_limit_hits",
    "agent_step_limit_exits",
    "agent_prompt_tokens",
    "agent_cached_prompt_tokens",
    "agent_uncached_prompt_tokens",
    "prompt_cache_hit_ratio",
    "cache_telemetry_calls",
    "usage_telemetry_complete",
    "cache_telemetry_complete",
    "agent_completion_tokens",
    "agent_total_tokens",
    "context_observations_tracked",
    "context_observations_untracked",
    "context_prompt_manifests",
    "context_full_history_tokens",
    "context_selected_history_tokens",
    "context_estimated_tokens_saved",
    "context_history_retention_ratio",
    "context_compact_prompt_entries",
    "context_committed_observations",
    "context_view_switches",
    "context_selection_changed_entries",
    "context_budget_overflow_tokens",
    "context_budget_overflow_prompts",
    "runner_exit_code",
    "runner_completed",
    "wall_time_seconds",
    "vllm_metrics_available",
    "vllm_metrics_isolated",
    "vllm_prompt_tokens",
    "vllm_prefix_cache_queries",
    "vllm_prefix_cache_hits",
    "vllm_prefix_cache_hit_ratio",
    "vllm_successful_requests",
    "vllm_ttft_seconds_total",
    "vllm_ttft_count",
    "vllm_ttft_seconds_mean",
    "vllm_prefill_seconds_total",
    "vllm_prefill_count",
    "vllm_prefill_seconds_mean",
    "vllm_e2e_latency_seconds_total",
    "vllm_e2e_latency_count",
    "vllm_e2e_latency_seconds_mean",
    "resolved",
    "graded",
    "resolve_rate",
    "agent_api_calls_delta_vs_reference",
    "agent_prompt_token_ratio_vs_reference",
    "agent_uncached_prompt_token_ratio_vs_reference",
    "wall_time_ratio_vs_reference",
    "vllm_prefix_cache_hit_ratio_delta_vs_reference",
    "vllm_prefill_time_ratio_vs_reference",
    "vllm_ttft_ratio_vs_reference",
    "vllm_e2e_latency_ratio_vs_reference",
    "resolve_rate_delta_vs_reference",
]


def write_summary(run_root: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    json_path = run_root / "summary.json"
    csv_path = run_root / "summary.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in FIELDS})
    return json_path, csv_path


def convert_predictions(source: Path, output: Path) -> int:
    payload = _read_json(source)
    rows = list(payload.values()) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("predictions must be a JSON object or array")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("each prediction must be an object")
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate agent-context SWE-Bench runs")
    subparsers = parser.add_subparsers(dest="command", required=True)
    summary = subparsers.add_parser("summary")
    summary.add_argument("--run-root", required=True)
    summary.add_argument("--reference-arm", default="R")
    convert = subparsers.add_parser("convert-preds")
    convert.add_argument("--input", required=True)
    convert.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "convert-preds":
        print(
            f"converted {convert_predictions(Path(args.input), Path(args.output))} predictions to {args.output}"
        )
        return 0
    root = Path(args.run_root).resolve()
    rows = summarize_run(root, reference_arm=args.reference_arm)
    json_path, csv_path = write_summary(root, rows)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    for row in rows:
        ratio = row["agent_prompt_token_ratio_vs_reference"]
        prompt_delta = "-" if ratio is None else f"{(ratio - 1.0) * 100:+.1f}%"
        cache_ratio = row["prompt_cache_hit_ratio"]
        cache_text = "n/a" if cache_ratio is None else f"{cache_ratio * 100:.1f}%"
        vllm_cache_ratio = row["vllm_prefix_cache_hit_ratio"]
        vllm_cache_text = "n/a" if vllm_cache_ratio is None else f"{vllm_cache_ratio * 100:.1f}%"
        print(
            f"{row['arm']}: tasks={row['trajectories']} calls={row['agent_api_calls']} "
            f"prompt_vs_R={prompt_delta} response_cache_hit={cache_text} "
            f"vllm_prefix_cache_hit={vllm_cache_text} "
            f"context_retention={row['context_history_retention_ratio']} "
            f"overflow={row['context_budget_overflow_tokens']} "
            f"wall={row['wall_time_seconds']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
