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
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    completion = int(usage.get("completion_tokens", 0) or 0)
    total = int(usage.get("total_tokens", 0) or prompt + completion)
    return prompt, completion, total


def _wall_time(path: Path) -> float | None:
    started = path / "started_at"
    ended = path / "ended_at"
    if not started.is_file() or not ended.is_file():
        return None
    start_time = datetime.fromisoformat(
        started.read_text(encoding="utf-8").strip().replace("Z", "+00:00")
    )
    end_time = datetime.fromisoformat(
        ended.read_text(encoding="utf-8").strip().replace("Z", "+00:00")
    )
    return max(0.0, (end_time - start_time).total_seconds())


def summarize_arm(path: Path) -> dict[str, Any]:
    trajectories = sorted(path.rglob("*.traj.json"))
    predictions: Any = {}
    if (path / "preds.json").is_file():
        predictions = _read_json(path / "preds.json")
    if not isinstance(predictions, dict):
        predictions = {}

    status_counts: Counter[str] = Counter()
    exit_statuses: Counter[str] = Counter()
    prompt_tokens = completion_tokens = total_tokens = 0
    agent_api_calls = 0
    posterior_calls = 0
    posterior_forwards = 0
    scoring_prompt_tokens = 0
    original_tokens = 0
    kept_tokens = 0

    for trajectory in trajectories:
        payload = _read_json(trajectory)
        if not isinstance(payload, dict):
            continue
        info = payload.get("info", {})
        if isinstance(info, dict):
            exit_statuses[str(info.get("exit_status", "unknown"))] += 1
        usage_calls = 0
        for message in payload.get("messages", []):
            if not isinstance(message, dict):
                continue
            if message.get("role") == "assistant":
                prompt, completion, total = _usage(message)
                if prompt or completion or total:
                    usage_calls += 1
                    prompt_tokens += prompt
                    completion_tokens += completion
                    total_tokens += total
            stats = message.get("posterior_pruned_stats")
            if not isinstance(stats, dict):
                continue
            status = str(stats.get("status", "unknown"))
            status_counts[status] += 1
            if status != "skipped":
                posterior_calls += 1
            posterior_forwards += int(stats.get("model_forward_count", 0) or 0)
            scoring_prompt_tokens += int(stats.get("scoring_prompt_tokens", 0) or 0)
            original_tokens += int(stats.get("original_estimated_tokens", 0) or 0)
            kept_tokens += int(stats.get("kept_estimated_tokens", 0) or 0)
        agent_api_calls += usage_calls

    exit_code = None
    if (path / "exit_code").is_file():
        exit_code = int((path / "exit_code").read_text(encoding="utf-8").strip())
    return {
        "arm": path.name,
        "predictions": len(predictions),
        "trajectories": len(trajectories),
        "submitted": exit_statuses.get("Submitted", 0),
        "agent_api_calls": agent_api_calls,
        "agent_prompt_tokens": prompt_tokens,
        "agent_completion_tokens": completion_tokens,
        "agent_total_tokens": total_tokens,
        "posterior_calls": posterior_calls,
        "posterior_model_forwards": posterior_forwards,
        "posterior_scoring_prompt_tokens": scoring_prompt_tokens,
        "posterior_accepted": status_counts.get("accepted", 0),
        "posterior_rejected": status_counts.get("rejected", 0),
        "posterior_skipped": status_counts.get("skipped", 0),
        "posterior_errors": status_counts.get("error", 0) + status_counts.get("client_error", 0),
        "original_observation_estimated_tokens": original_tokens,
        "kept_observation_estimated_tokens": kept_tokens,
        "observation_retention_ratio": (kept_tokens / original_tokens if original_tokens else None),
        "runner_exit_code": exit_code,
        "wall_time_seconds": _wall_time(path),
        "exit_statuses": dict(sorted(exit_statuses.items())),
        "posterior_statuses": dict(sorted(status_counts.items())),
        "resolved": None,
        "graded": None,
        "resolve_rate": None,
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
    root = run_root / "grade" / arm
    if not root.is_dir():
        return None
    candidates = sorted(
        (path for path in root.rglob("*.json") if path.name not in {"preds.json", "preds.jsonl"}),
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


def summarize_run(run_root: Path) -> list[dict[str, Any]]:
    arms_root = run_root / "arms"
    if not arms_root.is_dir():
        raise FileNotFoundError(f"arms directory does not exist: {arms_root}")
    rows = [summarize_arm(path) for path in sorted(arms_root.iterdir()) if path.is_dir()]
    for row in rows:
        grader = _grader_metrics(run_root, row["arm"])
        if grader:
            row["resolved"], row["graded"] = grader
            row["resolve_rate"] = row["resolved"] / row["graded"] if row["graded"] else None
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
    "posterior_calls",
    "posterior_model_forwards",
    "posterior_scoring_prompt_tokens",
    "posterior_accepted",
    "posterior_rejected",
    "posterior_skipped",
    "posterior_errors",
    "original_observation_estimated_tokens",
    "kept_observation_estimated_tokens",
    "observation_retention_ratio",
    "runner_exit_code",
    "wall_time_seconds",
    "resolved",
    "graded",
    "resolve_rate",
]


def write_summary(run_root: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    json_path = run_root / "summary.json"
    csv_path = run_root / "summary.csv"
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
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
    parser = argparse.ArgumentParser(description="Aggregate posterior SWE-Bench runs")
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
        root = Path(args.run_root).resolve()
        rows = summarize_run(root)
        json_path, csv_path = write_summary(root, rows)
        print(f"wrote {json_path}")
        print(f"wrote {csv_path}")
        for row in rows:
            retention = row["observation_retention_ratio"]
            retention_text = "-" if retention is None else f"{retention:.3f}"
            resolve_rate = row["resolve_rate"]
            resolve_text = "-" if resolve_rate is None else f"{resolve_rate:.3f}"
            print(
                f"{row['arm']}: tasks={row['trajectories']} "
                f"agent_calls={row['agent_api_calls']} "
                f"posterior_forwards={row['posterior_model_forwards']} "
                f"retention={retention_text} resolve_rate={resolve_text}"
            )
        return 0
    count = convert_predictions(Path(args.input), Path(args.output))
    print(f"converted {count} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
