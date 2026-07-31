from __future__ import annotations

import inspect
import json

from zero_forward_pruning.mini_adapter.hook import (
    SWE_PRUNER_SINGLE_MODE,
    assert_mini_compatible,
)


def main() -> int:
    try:
        from minisweagent.agents.default import DefaultAgent
    except ImportError as exc:
        raise SystemExit(
            "cannot import minisweagent; use MINI_SWE_PYTHON from its existing environment"
        ) from exc
    mini_mode = assert_mini_compatible(DefaultAgent)
    try:
        from minisweagent.run.extra import swebench  # type: ignore[attr-defined]

        runner = "minisweagent.run.extra.swebench"
    except ImportError:
        try:
            from minisweagent.run.benchmarks import swebench

            runner = "minisweagent.run.benchmarks.swebench"
        except ImportError as exc:
            raise SystemExit("cannot import a supported mini-swe-agent SWE-Bench runner") from exc
    if not hasattr(swebench, "app"):
        raise SystemExit("mini-swe-agent SWE-Bench module has no Typer app")
    if mini_mode == SWE_PRUNER_SINGLE_MODE:
        try:
            runner_parameters = inspect.signature(swebench.main).parameters
        except (TypeError, ValueError) as exc:
            raise SystemExit("cannot inspect SWE-Pruner eval runner main()") from exc
        missing = sorted({"pruner_url", "disable_pruner"} - set(runner_parameters))
        if missing:
            raise SystemExit(
                "SWE-Pruner eval runner contract changed; missing parameters: "
                + ", ".join(missing)
            )
        hook = "DefaultAgent.execute_action(self, action)"
        legacy_pruner_strategy = "empty-config-preserve-context-focus-question"
    else:
        hook = "DefaultAgent.execute_actions(self, message)"
        legacy_pruner_strategy = "runner-disable-flag-if-supported"
    print(
        json.dumps(
            {
                "status": "compatible",
                "mini_mode": mini_mode,
                "hook": hook,
                "runner": runner,
                "legacy_pruner_strategy": legacy_pruner_strategy,
                "model_forward_count": 0,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
