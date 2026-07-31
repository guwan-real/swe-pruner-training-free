from __future__ import annotations

import inspect
import os
import sys
from typing import Any

from zero_forward_pruning.mini_adapter.hook import (
    SWE_PRUNER_SINGLE_MODE,
    detect_installed_mini_mode,
    install_hook,
)


def _has_option(name: str) -> bool:
    return any(argument == name or argument.startswith(f"{name}=") for argument in sys.argv[1:])


def _configure_legacy_pruner(main_function: Any, mini_mode: str) -> None:
    try:
        parameters = inspect.signature(main_function).parameters
    except (TypeError, ValueError):
        return
    if _has_option("--pruner-url"):
        raise SystemExit(
            "--pruner-url activates the legacy hook; use ZERO_FORWARD_PRUNER_URL instead"
        )
    if mini_mode == SWE_PRUNER_SINGLE_MODE:
        if _has_option("--disable-pruner"):
            raise SystemExit(
                "do not pass --disable-pruner to the SWE-Pruner eval fork: it replaces "
                "the context_focus_question prompts; the generated shared config already "
                "removes agent.pruner for baseline and zero-forward arms"
            )
        # The official fork's runner restores an empty pruner: {} when this flag is
        # absent. AgentConfig treats that value as false, so it preserves the focus
        # prompts without constructing a legacy PrunerClient.
        return
    if "disable_pruner" in parameters and not _has_option("--disable-pruner"):
        sys.argv.append("--disable-pruner")


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


def main() -> int:
    mini_mode = detect_installed_mini_mode()
    installed = install_hook()
    allow_baseline = os.getenv("ZERO_FORWARD_ALLOW_BASELINE", "0") == "1"
    if not installed and not allow_baseline:
        raise SystemExit("ZERO_FORWARD_PRUNER_URL is required for non-baseline arms")
    swebench = _load_runner()
    _configure_legacy_pruner(swebench.main, mini_mode)
    swebench.app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
