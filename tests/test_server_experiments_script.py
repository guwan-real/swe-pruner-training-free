from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_server_experiments.sh"


def _make_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_server_script_help() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "preflight|launch|smoke|status|results|grade|stop" in completed.stdout
    assert "active uv/venv" in completed.stdout
    assert "never falls back" in completed.stdout
    assert "AGENT_STEP_LIMIT=100" in completed.stdout
    source = SCRIPT.read_text(encoding="utf-8")
    assert '--step-limit "$AGENT_STEP_LIMIT"' in source


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
            "SKIP_PREFLIGHT": "1",
            "PYTHON_BIN": sys.executable,
            "WORK_DIR": str(work_dir),
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
    assert "arm=baseline" in completed.stderr
    assert "ir_ast_hybrid_keep50" in completed.stderr
    assert "replay" not in completed.stderr.lower()
    assert "dry-run complete" in completed.stderr
    assert not work_dir.exists()


def test_preflight_uses_existing_mini_agent_environment(tmp_path: Path) -> None:
    fake_commands = tmp_path / "commands"
    fake_commands.mkdir()
    docker_root = tmp_path / "docker-root"
    docker_root.mkdir()
    _make_executable(
        fake_commands / "docker",
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'if [[ "${1:-}" == info && "${2:-}" == --format ]]; then',
                f"  printf '%s\\n' {str(docker_root)!r}",
                "fi",
            ]
        ),
    )

    mini_bin = tmp_path / "mini-agent" / "bin"
    mini_bin.mkdir(parents=True)
    _make_executable(
        mini_bin / "mini-extra",
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'if [[ "${1:-}" == swebench && "${2:-}" == --help ]]; then',
                "  printf '%s\\n' '--pruner-url --disable-pruner --slice'",
                "  exit 0",
                "fi",
                "exit 2",
            ]
        ),
    )
    (mini_bin / "python").symlink_to(sys.executable)

    fake_modules = tmp_path / "fake-modules"
    (fake_modules / "minisweagent" / "agents").mkdir(parents=True)
    (fake_modules / "minisweagent" / "utils").mkdir(parents=True)
    for package in (
        fake_modules / "minisweagent",
        fake_modules / "minisweagent" / "agents",
        fake_modules / "minisweagent" / "utils",
    ):
        (package / "__init__.py").write_text("", encoding="utf-8")
    (fake_modules / "minisweagent" / "agents" / "default.py").write_text(
        "\n".join(
            [
                "from dataclasses import dataclass",
                "@dataclass",
                "class AgentConfig:",
                "    pruner: object | None = None",
                "    step_limit: int = 0",
            ]
        ),
        encoding="utf-8",
    )
    (fake_modules / "minisweagent" / "utils" / "pruner.py").write_text(
        "\n".join(
            [
                "class PrunerRequest:",
                "    model_fields = {'query': None, 'code': None, 'threshold': None}",
                "class PruneResponse:",
                "    model_fields = {",
                "        'pruned_code': None,",
                "        'origin_token_cnt': None,",
                "        'left_token_cnt': None,",
                "        'model_input_token_cnt': None,",
                "    }",
            ]
        ),
        encoding="utf-8",
    )

    base_config = tmp_path / "pruning.yaml"
    base_config.write_text(
        yaml.safe_dump(
            {
                "model": {"model_name": "placeholder"},
                "agent": {
                    "system_template": "Return context_focus_question before tool use",
                    "pruner": {"url": "http://127.0.0.1:8016/prune"},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update(
        {
            "VIRTUAL_ENV": str(mini_bin.parent),
            "PATH": f"{mini_bin}:{fake_commands}:{env['PATH']}",
            "PYTHONPATH": f"{fake_modules}:{env.get('PYTHONPATH', '')}",
            "SKIP_CONDA": "1",
            "PYTHON_BIN": sys.executable,
            "WORK_DIR": str(tmp_path / "workspace"),
            "MINI_SWE_BASE_CONFIG": str(base_config),
            "VLLM_MODEL_ID": "Qwen3.5-27B",
            "MIN_FREE_DISK_GB": "0",
            "WARN_FREE_DISK_GB": "0",
        }
    )
    env.pop("MINI_EXTRA_BIN", None)
    env.pop("MINI_SWE_PYTHON", None)

    completed = subprocess.run(
        ["bash", str(SCRIPT), "preflight"],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert f"disabled active uv/venv: {mini_bin.parent}" in completed.stderr
    assert f"mini-swe-agent CLI: {mini_bin / 'mini-extra'}" in completed.stderr
    assert f"mini-swe-agent Python: {mini_bin / 'python'}" in completed.stderr
    assert f"pruning config: {base_config}" in completed.stderr
    assert "vLLM ready" in completed.stderr
    assert "preflight passed" in completed.stderr
