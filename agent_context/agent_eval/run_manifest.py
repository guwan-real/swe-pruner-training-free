from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _prediction_ids(payload: Any) -> list[str]:
    if isinstance(payload, dict):
        values = payload.items()
    elif isinstance(payload, list):
        values = enumerate(payload)
    else:
        return []
    ids: list[str] = []
    for key, value in values:
        if isinstance(value, dict):
            instance_id = value.get("instance_id") or value.get("task_id")
        else:
            instance_id = None
        if instance_id is None and isinstance(payload, dict):
            instance_id = key
        if instance_id is None:
            raise ValueError("list predictions must include instance_id or task_id")
        ids.append(str(instance_id))
    if len(ids) != len(set(ids)):
        raise ValueError("prediction task ids must be unique")
    return sorted(ids)


def _trajectory_id(arm_root: Path, trajectory: Path) -> str:
    relative = trajectory.relative_to(arm_root)
    if len(relative.parts) > 1:
        return relative.parts[0]
    suffix = ".traj.json"
    return relative.name[: -len(suffix)] if relative.name.endswith(suffix) else relative.stem


def finalize_manifest(manifest_path: Path, arms_root: Path) -> dict[str, Any]:
    payload = _read_json(manifest_path)
    if not isinstance(payload, dict):
        raise ValueError("run manifest must be a JSON object")
    arms = payload.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError("run manifest must select at least one arm")

    task_ids_by_arm: dict[str, list[str]] = {}
    trajectory_ids_by_arm: dict[str, list[str]] = {}
    trajectory_keys_by_arm: dict[str, list[str]] = {}
    for value in arms:
        arm = str(value)
        arm_root = arms_root / arm
        predictions_path = arm_root / "preds.json"
        prediction_ids = (
            _prediction_ids(_read_json(predictions_path)) if predictions_path.is_file() else []
        )
        trajectories = sorted(arm_root.rglob("*.traj.json")) if arm_root.is_dir() else []
        task_ids_by_arm[arm] = prediction_ids
        trajectory_ids_by_arm[arm] = sorted(_trajectory_id(arm_root, item) for item in trajectories)
        trajectory_keys_by_arm[arm] = [str(item.relative_to(arm_root)) for item in trajectories]

    reference_arm = str(payload.get("reference_arm", ""))
    reference_ids = task_ids_by_arm.get(reference_arm, [])
    task_sets_match = bool(reference_ids) and all(
        task_ids_by_arm[arm] == reference_ids and trajectory_ids_by_arm[arm] == reference_ids
        for arm in task_ids_by_arm
    )
    payload.update(
        {
            "finalized_at": datetime.now(timezone.utc).isoformat(),
            "task_ids": reference_ids,
            "task_count": len(reference_ids),
            "task_ids_by_arm": task_ids_by_arm,
            "trajectory_ids_by_arm": trajectory_ids_by_arm,
            "trajectory_keys_by_arm": trajectory_keys_by_arm,
            "task_set_status": "matched" if task_sets_match else "mismatch",
        }
    )
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    if not task_sets_match:
        raise ValueError("prediction and trajectory task ids differ across arms")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finalize an agent-context run manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--arms-root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = finalize_manifest(
        Path(args.manifest).expanduser().resolve(),
        Path(args.arms_root).expanduser().resolve(),
    )
    print(f"recorded {payload['task_count']} task ids")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
