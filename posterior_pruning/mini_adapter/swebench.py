from __future__ import annotations

import inspect
import sys

from posterior_pruning.mini_adapter.hook import install_hook


def _disable_legacy_pruner_if_supported(main_function: object) -> None:
    try:
        parameters = inspect.signature(main_function).parameters
    except (TypeError, ValueError):
        return
    if "disable_pruner" not in parameters:
        return
    if "--pruner-url" in sys.argv:
        raise SystemExit(
            "--pruner-url is the legacy pre-action hook and cannot be combined "
            "with posterior pruning"
        )
    if "--disable-pruner" not in sys.argv:
        sys.argv.append("--disable-pruner")


def main() -> int:
    install_hook()
    try:
        from minisweagent.run.extra import swebench
    except ImportError as exc:
        raise SystemExit(
            "cannot import minisweagent.run.extra.swebench; use the Python from "
            "the existing mini-swe-agent installation"
        ) from exc
    _disable_legacy_pruner_if_supported(swebench.main)
    swebench.app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
