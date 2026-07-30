from __future__ import annotations

import inspect
import os
import sys
from typing import Any

from zero_forward_pruning.mini_adapter.hook import install_hook


def _disable_legacy_pruner_if_supported(main_function: Any) -> None:
    try:
        parameters = inspect.signature(main_function).parameters
    except (TypeError, ValueError):
        return
    if "disable_pruner" not in parameters:
        return
    if "--pruner-url" in sys.argv:
        raise SystemExit(
            "--pruner-url activates the legacy hook; use ZERO_FORWARD_PRUNER_URL instead"
        )
    if "--disable-pruner" not in sys.argv:
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
    installed = install_hook()
    allow_baseline = os.getenv("ZERO_FORWARD_ALLOW_BASELINE", "0") == "1"
    if not installed and not allow_baseline:
        raise SystemExit("ZERO_FORWARD_PRUNER_URL is required for non-baseline arms")
    swebench = _load_runner()
    _disable_legacy_pruner_if_supported(swebench.main)
    swebench.app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
