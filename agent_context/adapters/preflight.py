from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from agent_context.adapters.mini_swe import assert_mini_compatible
from agent_context.config import AgentContextConfig
from agent_context.models import ObservationKind
from agent_context.registry import DEFAULT_COMPONENT_REGISTRY


def _load_config(path: Path) -> AgentContextConfig:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"agent-context config must be an object: {path}")
    config = AgentContextConfig.from_mapping(payload)
    DEFAULT_COMPONENT_REGISTRY.components(config)
    reference_kinds = [
        kind.value
        for kind in ObservationKind
        if config.views_for_kind(kind.value).include_reference_view
    ]
    if reference_kinds:
        raise ValueError(f"reference views are not wired into mini-swe-agent: {path}")
    return config


def _load_runner_name() -> str:
    try:
        from minisweagent.run.extra import swebench

        return swebench.__name__
    except ImportError:
        try:
            from minisweagent.run.benchmarks import swebench

            return swebench.__name__
        except ImportError as exc:
            raise RuntimeError("cannot import a supported mini-swe-agent SWE-Bench runner") from exc


def _check_runner_cli(config_path: Path) -> tuple[str, ...]:
    environment = os.environ.copy()
    environment.update(
        {
            "AGENT_CONTEXT_ENABLED": "1",
            "AGENT_CONTEXT_CONFIG": str(config_path),
            "POSTERIOR_HISTORY_ENABLED": "0",
            "ZERO_FORWARD_PRUNER_URL": "",
        }
    )
    completed = subprocess.run(
        [sys.executable, "-m", "agent_context.adapters.swebench", "--help"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    help_text = completed.stdout + completed.stderr
    required = (
        "--subset",
        "--split",
        "--output",
        "--workers",
        "--config",
        "--slice",
        "--filter",
    )
    missing = tuple(option for option in required if option not in help_text)
    if missing:
        raise RuntimeError("mini-swe-agent runner is missing CLI options: " + ", ".join(missing))
    return required


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate agent-context server contracts")
    parser.add_argument("--config", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from minisweagent.agents.default import AgentConfig, DefaultAgent

    mode = assert_mini_compatible(DefaultAgent)
    agent_fields = getattr(AgentConfig, "__dataclass_fields__", {})
    if "step_limit" not in agent_fields:
        raise SystemExit("mini-swe-agent AgentConfig has no step_limit field")
    configs: dict[str, dict[str, Any]] = {}
    config_paths: list[Path] = []
    for value in args.config:
        path = Path(value).expanduser().resolve()
        config_paths.append(path)
        config = _load_config(path)
        configs[path.stem] = {
            "timing": config.timing,
            "codec_profile": config.codec_profile,
            "signal_provider": config.signal_provider,
            "signal_strategy": config.signal_strategy,
            "planner_mode": config.planner.mode,
            "cache_policy": config.planner.cache_policy,
        }
    if not config_paths:
        raise SystemExit("at least one --config is required")
    runner_options = _check_runner_cli(config_paths[0])
    print(
        json.dumps(
            {
                "status": "passed",
                "mini_mode": mode,
                "runner": _load_runner_name(),
                "step_limit_field": True,
                "runner_options": runner_options,
                "configs": configs,
                "model_forward_count": 0,
                "llm_token_count": 0,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
