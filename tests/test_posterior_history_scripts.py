from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_posterior_launcher_is_sequential_and_does_not_start_a_pruner_service() -> None:
    source = (ROOT / "scripts" / "run_posterior_history_swebench.sh").read_text(encoding="utf-8")

    assert 'PARALLEL_ARMS="${PARALLEL_ARMS:-0}"' in source
    assert "posterior_history_pruning.mini_adapter.swebench" in source
    assert "POSTERIOR_HISTORY_ENABLED" in source
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
