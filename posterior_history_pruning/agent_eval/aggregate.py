from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _usage(message: dict[str, Any]) -> tuple[int, int, int]:
    extra = message.get("extra")
    response = extra.get("response", {}) if isinstance(extra, dict) else {}
    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    if not isinstance(usage, dict):
        return 0, 0, 0
    prompt = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    completion = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
    return prompt, completion, int(usage.get("total_tokens", 0) or prompt + completion)


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


def summarize_arm(path: Path) -> dict[str, Any]:
    trajectories = sorted(path.rglob("*.traj.json"))
    predictions: Any = _read_json(path / "preds.json") if (path / "preds.json").is_file() else {}
    predictions = predictions if isinstance(predictions, dict) else {}
    calls = prompt_tokens = completion_tokens = total_tokens = 0
    tracked = eligible = compacted = prompt_compactions = saved_tokens = 0
    origin_tokens = retained_tokens = 0
    statuses: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    exit_statuses: Counter[str] = Counter()
    reports: list[dict[str, Any]] = []
    for trajectory in trajectories:
        payload = _read_json(trajectory)
        if not isinstance(payload, dict):
            continue
        info = payload.get("info")
        if isinstance(info, dict):
            exit_statuses[str(info.get("exit_status", "unknown"))] += 1
        messages = payload.get("messages", [])
        if not isinstance(messages, list):
            continue
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") == "assistant":
                prompt, completion, total = _usage(message)
                if prompt or completion or total:
                    calls += 1
                    prompt_tokens += prompt
                    completion_tokens += completion
                    total_tokens += total
            stats = message.get("posterior_history_stats")
            if isinstance(stats, dict):
                tracked += 1
                status = str(stats.get("status", "unknown"))
                reason = str(stats.get("reason", "unknown"))
                statuses[status] += 1
                reasons[reason] += 1
                if stats.get("posterior_command"):
                    eligible += 1
                if status == "compacted":
                    compacted += 1
                    origin_tokens += int(stats.get("origin_token_cnt", 0) or 0)
                    retained_tokens += int(stats.get("left_token_cnt", 0) or 0)
                    prompt_compactions += int(stats.get("prompt_compaction_count", 0) or 0)
                    saved_tokens += int(stats.get("total_prompt_tokens_saved", 0) or 0)
            report = message.get("posterior_history_report")
            if isinstance(report, dict):
                reports.append(report)
    exit_code = None
    if (path / "exit_code").is_file():
        exit_code = int((path / "exit_code").read_text(encoding="utf-8").strip())
    return {
        "arm": path.name,
        "predictions": len(predictions),
        "trajectories": len(trajectories),
        "submitted": exit_statuses.get("Submitted", 0),
        "agent_api_calls": calls,
        "agent_prompt_tokens": prompt_tokens,
        "agent_completion_tokens": completion_tokens,
        "agent_total_tokens": total_tokens,
        "history_observations_tracked": tracked,
        "posterior_eligible_observations": eligible,
        "posterior_compacted_observations": compacted,
        "history_prompt_compactions": prompt_compactions,
        "estimated_history_tokens_saved": saved_tokens,
        "history_observation_retention_ratio": (
            retained_tokens / origin_tokens if origin_tokens else None
        ),
        "pruner_model_forwards": 0,
        "pruner_llm_tokens": 0,
        "runner_exit_code": exit_code,
        "wall_time_seconds": _wall_time(path),
        "exit_statuses": dict(sorted(exit_statuses.items())),
        "posterior_statuses": dict(sorted(statuses.items())),
        "posterior_reasons": dict(sorted(reasons.items())),
        "trajectory_reports": reports,
        "resolved": None,
        "graded": None,
        "resolve_rate": None,
    }


def _ratio(value: int | float | None, baseline: int | float | None) -> float | None:
    return value / baseline if value is not None and baseline not in (None, 0) else None


def _delta(value: int | float | None, baseline: int | float | None) -> int | float | None:
    return value - baseline if value is not None and baseline is not None else None


def _add_baseline_comparisons(rows: list[dict[str, Any]]) -> None:
    baseline = next((row for row in rows if row["arm"] == "baseline"), None)
    fields = {
        "agent_api_calls_delta_vs_baseline": ("agent_api_calls", _delta),
        "agent_prompt_tokens_delta_vs_baseline": ("agent_prompt_tokens", _delta),
        "agent_prompt_token_ratio_vs_baseline": ("agent_prompt_tokens", _ratio),
        "agent_total_tokens_delta_vs_baseline": ("agent_total_tokens", _delta),
        "agent_total_token_ratio_vs_baseline": ("agent_total_tokens", _ratio),
        "wall_time_seconds_delta_vs_baseline": ("wall_time_seconds", _delta),
        "wall_time_ratio_vs_baseline": ("wall_time_seconds", _ratio),
        "resolve_rate_delta_vs_baseline": ("resolve_rate", _delta),
    }
    for row in rows:
        comparable = (
            baseline is not None
            and baseline["trajectories"] > 0
            and row["trajectories"] == baseline["trajectories"]
        )
        for destination, (source, operation) in fields.items():
            row[destination] = (
                operation(row.get(source), baseline.get(source)) if comparable else None
            )


def summarize_run(run_root: Path) -> list[dict[str, Any]]:
    arms = run_root / "arms"
    if not arms.is_dir():
        raise FileNotFoundError(f"arms directory does not exist: {arms}")
    rows = [summarize_arm(path) for path in sorted(arms.iterdir()) if path.is_dir()]
    for row in rows:
        graded = _grader_metrics(run_root, row["arm"])
        if graded:
            row["resolved"], row["graded"] = graded
            row["resolve_rate"] = row["resolved"] / row["graded"] if row["graded"] else None
    _add_baseline_comparisons(rows)
    return rows


FIELDS = [
    "arm",
    "predictions",
    "trajectories",
    "submitted",
    "agent_api_calls",
    "agent_prompt_tokens",
    "agent_completion_tokens",
    "agent_total_tokens",
    "history_observations_tracked",
    "posterior_eligible_observations",
    "posterior_compacted_observations",
    "history_prompt_compactions",
    "estimated_history_tokens_saved",
    "history_observation_retention_ratio",
    "pruner_model_forwards",
    "pruner_llm_tokens",
    "runner_exit_code",
    "wall_time_seconds",
    "resolved",
    "graded",
    "resolve_rate",
    "agent_api_calls_delta_vs_baseline",
    "agent_prompt_tokens_delta_vs_baseline",
    "agent_prompt_token_ratio_vs_baseline",
    "agent_total_tokens_delta_vs_baseline",
    "agent_total_token_ratio_vs_baseline",
    "wall_time_seconds_delta_vs_baseline",
    "wall_time_ratio_vs_baseline",
    "resolve_rate_delta_vs_baseline",
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
    parser = argparse.ArgumentParser(description="Aggregate posterior-history SWE-Bench runs")
    subparsers = parser.add_subparsers(dest="command", required=True)
    summary = subparsers.add_parser("summary")
    summary.add_argument("--run-root", required=True)
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
    rows = summarize_run(root)
    json_path, csv_path = write_summary(root, rows)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    for row in rows:
        ratio = row["agent_prompt_token_ratio_vs_baseline"]
        prompt_delta = "-" if ratio is None else f"{(ratio - 1.0) * 100:+.1f}%"
        print(
            f"{row['arm']}: tasks={row['trajectories']} calls={row['agent_api_calls']} "
            f"history_compacted={row['posterior_compacted_observations']} "
            f"history_saved={row['estimated_history_tokens_saved']} "
            f"prompt_vs_baseline={prompt_delta}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
