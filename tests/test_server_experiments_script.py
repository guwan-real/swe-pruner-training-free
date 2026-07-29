from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_server_experiments.sh"


def test_server_script_help() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "launch|smoke|status|results" in completed.stdout
    assert "active uv/venv" in completed.stdout


def test_dry_run_removes_uv_environment_without_writing(tmp_path: Path) -> None:
    fake_venv = tmp_path / "uv-project" / ".venv"
    fake_bin = fake_venv / "bin"
    fake_bin.mkdir(parents=True)
    work_dir = tmp_path / "workspace"
    env = os.environ.copy()
    env.update(
        {
            "VIRTUAL_ENV": str(fake_venv),
            "PATH": f"{fake_bin}:{env['PATH']}",
            "SKIP_CONDA": "1",
            "PYTHON_BIN": sys.executable,
            "WORK_DIR": str(work_dir),
            "OLD_PROJECT_DIR": str(tmp_path / "missing-old-project"),
        }
    )

    completed = subprocess.run(
        ["bash", str(SCRIPT), "launch", "--dry-run"],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert f"disabled active uv/venv: {fake_venv}" in completed.stderr
    assert "falling back to bundled demo replay" in completed.stderr
    assert "dry-run complete" in completed.stderr
    assert not work_dir.exists()
