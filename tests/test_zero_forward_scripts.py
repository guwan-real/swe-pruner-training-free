from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_launcher_documents_five_zero_forward_arms() -> None:
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_zero_forward_swebench.sh"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    for arm in (
        "baseline",
        "safe_rules",
        "intent_ir",
        "intent_structure",
        "adaptive_evidence",
    ):
        assert arm in result.stdout
    assert "zero LLM forwards" in result.stdout


def test_zero_forward_package_has_no_model_scoring_endpoint() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in (ROOT / "zero_forward_pruning").rglob("*.py")
    )
    assert "prompt_logprobs" not in source
    assert "/chat/completions" not in source
    assert "VLLMActionScorer" not in source


def test_removed_posterior_runtime_files_are_absent() -> None:
    assert not list((ROOT / "posterior_pruning").rglob("*.py"))
    assert not (ROOT / "scripts" / "run_posterior_swebench.sh").exists()


def test_conda_setup_disables_inherited_uv_before_activation() -> None:
    source = (ROOT / "scripts" / "create_server_conda.sh").read_text(encoding="utf-8")
    disable_position = source.index("unset VIRTUAL_ENV")
    activate_position = source.index('conda activate "$ENV_NAME"')
    assert disable_position < activate_position
    assert "UV_PROJECT_ENVIRONMENT" in source


def test_base_config_discovery_suppresses_mini_startup_banner() -> None:
    source = (ROOT / "scripts" / "run_zero_forward_swebench.sh").read_text(encoding="utf-8")
    assert 'MSWEA_SILENT_STARTUP=1 PYTHONPATH="$REPO_ROOT" "$MINI_SWE_PYTHON_BIN"' in source
