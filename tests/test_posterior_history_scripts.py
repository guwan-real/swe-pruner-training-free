from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_posterior_launcher_is_sequential_and_does_not_start_a_pruner_service() -> None:
    source = (ROOT / "scripts" / "run_posterior_history_swebench.sh").read_text(encoding="utf-8")

    assert 'PARALLEL_ARMS="${PARALLEL_ARMS:-0}"' in source
    assert "posterior_history_pruning.mini_adapter.swebench" in source
    assert "POSTERIOR_HISTORY_ENABLED" in source
    assert 'AGENT_STEP_LIMIT="${AGENT_STEP_LIMIT:-100}"' in source
    assert '--step-limit "$AGENT_STEP_LIMIT"' in source
    assert "start_service" not in source
    assert "zero_forward_pruning" not in source


def test_posterior_launcher_keeps_baseline_and_posterior_arms_in_one_shared_config() -> None:
    source = (ROOT / "scripts" / "run_posterior_history_swebench.sh").read_text(encoding="utf-8")

    assert "generate_shared_config" in source
    assert "start_arm baseline adaptive" in source
    assert 'start_arm "posterior_$method" "$method"' in source
    assert 'POSTERIOR_HISTORY_ALLOW_BASELINE="$allow_baseline"' in source


def test_posterior_launcher_is_safe_under_errexit_and_nounset() -> None:
    source = (ROOT / "scripts" / "run_posterior_history_swebench.sh").read_text(encoding="utf-8")

    assert 'if [[ -z "$active_venv" ]]; then return 0; fi' in source
    assert '[[ -n "$active_venv" ]] || return' not in source
    assert 'local arm="$1" method="$2"\n  local arm_dir="$RUN_ROOT/arms/$arm"' in source
    assert 'method="$2" arm_dir="$RUN_ROOT/arms/$arm"' not in source


def test_explicit_environment_overrides_server_profile(tmp_path: Path) -> None:
    profile = tmp_path / "posterior.env"
    profile.write_text(
        "POSTERIOR_MIN_INPUT_TOKENS=1500\nTASK_SLICE=0:5\nSKIP_BASELINE=0\nAGENT_STEP_LIMIT=0\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "SERVER_PROFILE": str(profile),
            "POSTERIOR_MIN_INPUT_TOKENS": "500",
            "TASK_SLICE": "0:20",
            "SKIP_BASELINE": "1",
            "RUN_TAG": "precedence_probe",
            "AGENT_STEP_LIMIT": "100",
        }
    )

    completed = subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_posterior_history_swebench.sh"), "config"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert "POSTERIOR_MIN_INPUT_TOKENS=500" in completed.stdout
    assert "TASK_SLICE=0:20" in completed.stdout
    assert "SKIP_BASELINE=1" in completed.stdout
    assert "RUN_TAG=precedence_probe" in completed.stdout
    assert "AGENT_STEP_LIMIT=100" in completed.stdout


def test_threshold_sweep_is_posterior_only_and_serial() -> None:
    launcher = (ROOT / "scripts" / "run_posterior_history_swebench.sh").read_text(encoding="utf-8")
    sweep = (ROOT / "scripts" / "run_posterior_threshold_sweep.sh").read_text(encoding="utf-8")

    assert 'if [[ "$SKIP_BASELINE" == "1" ]]' in launcher
    assert '"min_input_tokens": int(min_input)' in launcher
    assert '"baseline_included": skip_baseline != "1"' in launcher
    assert "POSTERIOR_THRESHOLDS:-1000,500" in sweep
    assert "SKIP_BASELINE=1" in sweep
    assert "PARALLEL_ARMS=0" in sweep
    assert "POSTERIOR_HISTORY_METHODS=adaptive" in sweep
    assert "AGENT_STEP_LIMIT=100" in sweep
    assert 'TASK_SLICE="$task_slice"' in sweep
    assert 'bash "$LAUNCHER" launch' in sweep
