from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_context.agent_eval.run_manifest import finalize_manifest


def _write_arm(root: Path, arm: str, task_ids: list[str]) -> None:
    arm_root = root / "arms" / arm
    predictions = {}
    for task_id in task_ids:
        trajectory = arm_root / task_id / f"{task_id}.traj.json"
        trajectory.parent.mkdir(parents=True, exist_ok=True)
        trajectory.write_text(json.dumps({"info": {"exit_status": "Submitted"}}), encoding="utf-8")
        predictions[task_id] = {"instance_id": task_id, "model_patch": "patch"}
    (arm_root / "preds.json").write_text(json.dumps(predictions), encoding="utf-8")


def test_finalize_manifest_records_exact_task_ids_for_every_arm(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"arms": ["R", "C"], "reference_arm": "R"}),
        encoding="utf-8",
    )
    _write_arm(tmp_path, "R", ["django__django-1", "pytest-dev__pytest-2"])
    _write_arm(tmp_path, "C", ["django__django-1", "pytest-dev__pytest-2"])

    payload = finalize_manifest(manifest, tmp_path / "arms")

    assert payload["task_set_status"] == "matched"
    assert payload["task_count"] == 2
    assert payload["task_ids"] == ["django__django-1", "pytest-dev__pytest-2"]
    assert payload["task_ids_by_arm"]["C"] == payload["task_ids"]


def test_finalize_manifest_records_then_rejects_mismatched_task_sets(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"arms": ["R", "C"], "reference_arm": "R"}),
        encoding="utf-8",
    )
    _write_arm(tmp_path, "R", ["task-1", "task-2"])
    _write_arm(tmp_path, "C", ["task-1"])

    with pytest.raises(ValueError, match="task ids differ"):
        finalize_manifest(manifest, tmp_path / "arms")

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["task_set_status"] == "mismatch"
    assert payload["task_ids_by_arm"]["C"] == ["task-1"]
