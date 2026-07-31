from __future__ import annotations

import inspect
import os
import sys
from typing import Any

from posterior_history_pruning.mini_adapter.hook import (
    SWE_PRUNER_POSTERIOR_MODE,
    detect_installed_mini_mode,
    install_hook,
)


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


def _configure_legacy_pruner(main_function: Any, mini_mode: str) -> None:
    try:
        parameters = inspect.signature(main_function).parameters
    except (TypeError, ValueError):
        return
    if _has_option("--pruner-url"):
        raise SystemExit("--pruner-url activates the legacy SWE-Pruner client")
    if mini_mode == SWE_PRUNER_POSTERIOR_MODE:
        if _has_option("--disable-pruner"):
            raise SystemExit(
                "do not pass --disable-pruner to the SWE-Pruner eval fork: it changes "
                "the context_focus_question prompt and breaks baseline parity"
            )
        return
    if "disable_pruner" in parameters and not _has_option("--disable-pruner"):
        sys.argv.append("--disable-pruner")


def main() -> int:
    mini_mode = detect_installed_mini_mode()
    installed = install_hook()
    allow_baseline = os.getenv("POSTERIOR_HISTORY_ALLOW_BASELINE", "0") == "1"
    if not installed and not allow_baseline:
        raise SystemExit("POSTERIOR_HISTORY_ENABLED=1 is required for non-baseline arms")
    swebench = _load_runner()
    _configure_legacy_pruner(swebench.main, mini_mode)
    swebench.app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
