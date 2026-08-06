from __future__ import annotations

import os
import sys

from agent_context.adapters.mini_swe import config_from_env, install_hook


def _has_option(name: str) -> bool:
    return any(argument == name or argument.startswith(f"{name}=") for argument in sys.argv[1:])


def _load_runner():
    try:
        from minisweagent.run.extra import swebench  # type: ignore[attr-defined]

        return swebench
    except ImportError:
        try:
            from minisweagent.run.benchmarks import swebench

            return swebench
        except ImportError as exc:
            raise SystemExit("cannot import a supported mini-swe-agent SWE-Bench runner") from exc


def _guard_legacy_pruner_options() -> None:
    if _has_option("--pruner-url"):
        raise SystemExit("--pruner-url activates the legacy SWE-Pruner client")
    if _has_option("--disable-pruner"):
        raise SystemExit(
            "do not pass --disable-pruner: it changes the context-focus prompt and breaks arm parity"
        )


def main() -> int:
    config = config_from_env()
    if config is None:
        raise SystemExit("AGENT_CONTEXT_ENABLED=1 is required")
    _guard_legacy_pruner_options()
    if os.getenv("POSTERIOR_HISTORY_ENABLED", "0") == "1":
        raise SystemExit("disable POSTERIOR_HISTORY_ENABLED; history hooks cannot be stacked")
    if os.getenv("ZERO_FORWARD_PRUNER_URL"):
        raise SystemExit("disable ZERO_FORWARD_PRUNER_URL; context hooks cannot be stacked")
    if not install_hook(config):
        raise SystemExit("failed to install agent-context hook")
    runner = _load_runner()
    runner.app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
