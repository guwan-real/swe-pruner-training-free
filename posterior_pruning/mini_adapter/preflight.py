from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

from posterior_pruning.mini_adapter.hook import assert_mini_compatible


def _probe_query_appends_assistant(default_agent_class: type) -> None:
    class FakeModel:
        n_calls = 0
        cost = 0.0

        def query(self, messages):
            return {"content": "```bash\npwd\n```", "extra": {}}

    agent = object.__new__(default_agent_class)
    agent.model = FakeModel()
    agent.messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
    ]
    agent.config = SimpleNamespace(
        step_limit=0,
        cost_limit=0.0,
        action_regex=r"```bash\s*\n(.*?)\n```",
        format_error_template="format error",
    )
    agent.extra_template_vars = {}
    response = default_agent_class.query(agent)
    if response.get("content") != "```bash\npwd\n```":
        raise RuntimeError("DefaultAgent.query did not return the model response")
    if agent.messages[-1].get("role") != "assistant":
        raise RuntimeError("DefaultAgent.query no longer appends the assistant message")


def main() -> int:
    try:
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.run.extra import swebench
    except ImportError as exc:
        raise SystemExit(f"mini-swe-agent import failed: {exc}") from exc
    assert_mini_compatible(DefaultAgent)
    _probe_query_appends_assistant(DefaultAgent)
    parameters = inspect.signature(swebench.main).parameters
    required = {"subset", "split", "output", "workers", "config_spec"}
    missing = sorted(required - set(parameters))
    if missing:
        raise SystemExit(f"mini swebench CLI is missing parameters: {', '.join(missing)}")
    print(
        json.dumps(
            {
                "status": "ok",
                "default_agent": f"{DefaultAgent.__module__}.{DefaultAgent.__name__}",
                "swebench_module": swebench.__file__,
                "supports_slice": "slice_spec" in parameters,
                "supports_legacy_disable": "disable_pruner" in parameters,
                "adapter_timing": "after-current-query-before-future-query",
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
