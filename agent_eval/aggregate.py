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


def _usage_from_message(message: dict[str, Any]) -> tuple[int, int, int]:
    response = message.get("extra", {}).get("response", {})
    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    completion = int(usage.get("completion_tokens", 0) or 0)
    total = int(usage.get("total_tokens", 0) or prompt + completion)
    return prompt, completion, total


def summarize_arm(path: Path) -> dict[str, Any]:
    trajectories = sorted(path.rglob("*.traj.json"))
    preds_path = path / "preds.json"
    predictions = _read_json(preds_path) if preds_path.is_file() else {}
    if not isinstance(predictions, dict):
        predictions = {}

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    api_calls = 0
    prune_calls = 0
    prune_errors = 0
    original_observation_tokens = 0
    kept_observation_tokens = 0
    exit_statuses: Counter[str] = Counter()
    calls_per_trajectory: list[int] = []

    for trajectory in trajectories:
        payload = _read_json(trajectory)
        info = payload.get("info", {})
        exit_statuses[str(info.get("exit_status", "unknown"))] += 1
        model_stats = info.get("model_stats", {})
        trajectory_api_calls = int(model_stats.get("api_calls", 0) or 0)
        usage_calls = 0
        for message in payload.get("messages", []):
            if not isinstance(message, dict):
                continue
            if message.get("role") == "assistant":
                prompt, completion, total = _usage_from_message(message)
                if prompt or completion or total:
                    usage_calls += 1
                    prompt_tokens += prompt
                    completion_tokens += completion
                    total_tokens += total
            stats = message.get("pruned_stats")
            if isinstance(stats, dict):
                prune_calls += 1
                original_observation_tokens += int(stats.get("origin_token_cnt", 0) or 0)
                kept_observation_tokens += int(stats.get("left_token_cnt", 0) or 0)
                if "[Pruner Error]" in str(message.get("content", "")):
                    prune_errors += 1
        calls_for_trajectory = max(trajectory_api_calls, usage_calls)
        calls_per_trajectory.append(calls_for_trajectory)
        api_calls += calls_for_trajectory

    retention = (
        kept_observation_tokens / original_observation_tokens
        if original_observation_tokens
        else None
    )
    exit_code_path = path / "exit_code"
    runner_exit_code = (
        int(exit_code_path.read_text(encoding="utf-8").strip())
        if exit_code_path.is_file()
        else None
    )
    wall_time_seconds = None
    started_path = path / "started_at"
    ended_path = path / "ended_at"
    if started_path.is_file() and ended_path.is_file():
        started = datetime.fromisoformat(
            started_path.read_text(encoding="utf-8").strip().replace("Z", "+00:00")
        )
        ended = datetime.fromisoformat(
            ended_path.read_text(encoding="utf-8").strip().replace("Z", "+00:00")
        )
        wall_time_seconds = max(0.0, (ended - started).total_seconds())
    return {
        "arm": path.name,
        "predictions": len(predictions),
        "trajectories": len(trajectories),
        "submitted": exit_statuses.get("Submitted", 0),
        "api_calls": api_calls,
        "api_calls_mean_per_task": (
            api_calls / len(calls_per_trajectory) if calls_per_trajectory else None
        ),
        "api_calls_max_per_task": max(calls_per_trajectory, default=0),
        "agent_step_limit_hits": sum(value >= 100 for value in calls_per_trajectory),
        "agent_step_limit_exits": exit_statuses.get("LimitsExceeded", 0),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "prune_calls": prune_calls,
        "prune_errors": prune_errors,
        "original_observation_tokens": original_observation_tokens,
        "kept_observation_tokens": kept_observation_tokens,
        "observation_retention_ratio": retention,
        "runner_exit_code": runner_exit_code,
        "wall_time_seconds": wall_time_seconds,
        "exit_statuses": dict(sorted(exit_statuses.items())),
        "resolve_rate": None,
        "resolved": None,
        "graded": None,
        "note": "Resolve Rate remains null until the official SWE-Bench grader is run.",
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


def _grader_metrics(run_root: Path, arm: str) -> tuple[int, int] | None:
    grade_root = run_root / "grade" / arm
    if not grade_root.is_dir():
        return None
    candidates = sorted(
        (
            path
            for path in grade_root.rglob("*.json")
            if path.name not in {"preds.json", "preds.jsonl"}
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        try:
            payload = _read_json(candidate)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        resolved = _count_value(
            payload,
            "resolved_instances",
            "resolved_ids",
            "resolved",
        )
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


def summarize_run(run_root: Path) -> list[dict[str, Any]]:
    arms_root = run_root / "arms"
    if not arms_root.is_dir():
        raise FileNotFoundError(f"arms directory does not exist: {arms_root}")
    rows = [summarize_arm(path) for path in sorted(arms_root.iterdir()) if path.is_dir()]
    for row in rows:
        grader = _grader_metrics(run_root, row["arm"])
        if grader is None:
            continue
        resolved, graded = grader
        row["resolved"] = resolved
        row["graded"] = graded
        row["resolve_rate"] = resolved / graded if graded else None
        row["note"] = "Resolve Rate read from the official SWE-Bench grader report."
    return rows


def write_summary(run_root: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    json_path = run_root / "summary.json"
    csv_path = run_root / "summary.csv"
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fields = [
        "arm",
        "predictions",
        "trajectories",
        "submitted",
        "api_calls",
        "api_calls_mean_per_task",
        "api_calls_max_per_task",
        "agent_step_limit_hits",
        "agent_step_limit_exits",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prune_calls",
        "prune_errors",
        "original_observation_tokens",
        "kept_observation_tokens",
        "observation_retention_ratio",
        "runner_exit_code",
        "wall_time_seconds",
        "resolved",
        "graded",
        "resolve_rate",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    return json_path, csv_path


def convert_predictions(source: Path, output: Path) -> int:
    payload = _read_json(source)
    if isinstance(payload, dict):
        rows = list(payload.values())
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError("predictions must be a JSON object or array")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("each prediction must be an object")
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate mini-swe-agent experiment outputs")
    subparsers = parser.add_subparsers(dest="command", required=True)
    summary = subparsers.add_parser("summary")
    summary.add_argument("--run-root", required=True)
    convert = subparsers.add_parser("convert-preds")
    convert.add_argument("--input", required=True)
    convert.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "summary":
        run_root = Path(args.run_root).resolve()
        rows = summarize_run(run_root)
        json_path, csv_path = write_summary(run_root, rows)
        print(f"wrote {json_path}")
        print(f"wrote {csv_path}")
        for row in rows:
            retention = row["observation_retention_ratio"]
            retention_text = "-" if retention is None else f"{retention:.3f}"
            resolve_rate = row["resolve_rate"]
            resolve_text = "-" if resolve_rate is None else f"{resolve_rate:.3f}"
            print(
                f"{row['arm']}: tasks={row['trajectories']} "
                f"api_calls={row['api_calls']} "
                f"max_calls_per_task={row['api_calls_max_per_task']} "
                f"limit_hits={row['agent_step_limit_hits']} retention={retention_text} "
                f"resolve_rate={resolve_text}"
            )
        return 0
    count = convert_predictions(Path(args.input), Path(args.output))
    print(f"converted {count} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
