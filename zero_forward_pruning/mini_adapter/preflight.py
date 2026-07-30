from __future__ import annotations

import json

from zero_forward_pruning.mini_adapter.hook import assert_mini_compatible


def main() -> int:
    try:
        from minisweagent.agents.default import DefaultAgent
    except ImportError as exc:
        raise SystemExit(
            "cannot import minisweagent; use MINI_SWE_PYTHON from its existing environment"
        ) from exc
    assert_mini_compatible(DefaultAgent)
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
    print(
        json.dumps(
            {
                "status": "compatible",
                "hook": "DefaultAgent.execute_actions(self, message)",
                "runner": runner,
                "model_forward_count": 0,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
