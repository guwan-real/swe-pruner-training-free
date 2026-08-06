from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_agent_context_swebench.sh"
ARM_RUNNER = ROOT / "scripts" / "run_agent_context_arm.sh"


def test_launcher_defaults_to_serial_arms_and_collects_cache_metrics() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert 'AGENT_CONTEXT_ARMS="${AGENT_CONTEXT_ARMS:-R,D,E,F,C}"' in source
    assert 'AGENT_CONTEXT_REFERENCE_ARM="${AGENT_CONTEXT_REFERENCE_ARM:-R}"' in source
    assert 'PARALLEL_ARMS="${PARALLEL_ARMS:-0}"' in source
    assert "PARALLEL_ARMS=1 cannot produce isolated per-arm vLLM metrics" in source
    assert "AGENT_CONTEXT_CONFIG" in source
    assert "POSTERIOR_HISTORY_ENABLED=0" in source
    assert "snapshot_metrics" in source
    assert "vllm_metrics_before.prom" in source
    assert "vllm_metrics_after.prom" in source
    assert "check_metrics_endpoint" in source
    assert 'check_disk_path "runs" "$RUNS_DIR"' in source
    assert "R|D|E|F|F0|C" not in source
    assert 'payload["planner"]' not in source
    assert 'AGENT_CONTEXT_ARMS="R"' not in source
    assert "start_service" not in source
    assert "resolve_swebench_dataset_name" in source
    assert "repository worktree is dirty" in source
    assert '"git_worktree_clean"' in source


def test_explicit_environment_overrides_agent_context_profile(tmp_path: Path) -> None:
    profile = tmp_path / "agent-context.env"
    profile.write_text(
        "TASK_SLICE=0:5\nAGENT_CONTEXT_ARMS=R,D\nAGENT_CONTEXT_REFERENCE_ARM=D\n"
        "SMOKE_ARM=D\nPARALLEL_ARMS=1\nAGENT_STEP_LIMIT=0\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "SERVER_PROFILE": str(profile),
            "TASK_SLICE": "0:30",
            "AGENT_CONTEXT_ARMS": "R,E,F",
            "AGENT_CONTEXT_REFERENCE_ARM": "E",
            "SMOKE_ARM": "F",
            "PARALLEL_ARMS": "0",
            "AGENT_STEP_LIMIT": "100",
            "RUN_TAG": "agent_context_probe",
            "SWEBENCH_DATASET_NAME": "princeton-nlp/SWE-bench_Verified",
        }
    )

    completed = subprocess.run(
        ["bash", str(LAUNCHER), "config"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert "TASK_SLICE=0:30" in completed.stdout
    assert "AGENT_CONTEXT_ARMS=R,E,F" in completed.stdout
    assert "AGENT_CONTEXT_REFERENCE_ARM=E" in completed.stdout
    assert "PARALLEL_ARMS=0" in completed.stdout
    assert "RUN_TAG=agent_context_probe" in completed.stdout
    assert "SWEBENCH_DATASET_NAME=princeton-nlp/SWE-bench_Verified" in completed.stdout
    assert "RESOLVED_SWEBENCH_DATASET_NAME=princeton-nlp/SWE-bench_Verified" in completed.stdout
    assert "SMOKE_ARM=F" in completed.stdout


def test_launcher_does_not_allow_parallel_execution() -> None:
    environment = os.environ.copy()
    environment["PARALLEL_ARMS"] = "1"
    completed = subprocess.run(
        ["bash", str(LAUNCHER), "config"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert "PARALLEL_ARMS=1" in completed.stdout


def test_launcher_rejects_a_grading_dataset_that_differs_from_generation() -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "DATASET_SUBSET": "verified",
            "SWEBENCH_DATASET_NAME": "custom/dataset",
        }
    )
    completed = subprocess.run(
        ["bash", str(LAUNCHER), "config"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 2
    assert "does not match DATASET_SUBSET=verified" in completed.stderr


def test_launcher_rejects_dataset_subset_aliases_not_passed_to_the_runner() -> None:
    environment = os.environ.copy()
    environment["DATASET_SUBSET"] = "VERIFIED"
    completed = subprocess.run(
        ["bash", str(LAUNCHER), "config"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 2
    assert "unsupported DATASET_SUBSET=VERIFIED" in completed.stderr


def test_launcher_records_reproducibility_fields_and_grades_the_recorded_split() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert '"task_filter": task_filter' in source
    assert '"agent_workers": int(agent_workers)' in source
    assert "agent_context.agent_eval.run_manifest" in source
    assert '--dataset_name "$dataset_name" --split "$dataset_split"' in source


def test_arm_runner_records_pid_timestamps_and_failure_code(tmp_path: Path) -> None:
    arm_dir = tmp_path / "R"

    completed = subprocess.run(
        ["bash", str(ARM_RUNNER), str(arm_dir), "bash", "-c", "exit 7"],
        check=False,
    )

    assert completed.returncode == 7
    assert (arm_dir / "pid").read_text(encoding="utf-8").strip().isdigit()
    assert (arm_dir / "started_at").is_file()
    assert (arm_dir / "ended_at").is_file()
    assert (arm_dir / "exit_code").read_text(encoding="utf-8").strip() == "7"
