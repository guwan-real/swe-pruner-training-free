from __future__ import annotations

import re
from typing import Any

from agent_context.adapters.mini_swe import patch_agent
from agent_context.config import AgentContextConfig


def _source() -> str:
    values: list[str] = []
    for index in range(30):
        name = "resolve_model" if index == 15 else f"helper_{index}"
        values.extend(
            [
                f"def {name}(config):",
                f"    value = config.get('key_{index}')",
                "    if value is None:",
                f"        raise ValueError('key_{index}')",
                "    normalized = str(value).strip()",
                "    if not normalized:",
                "        return None",
                "    return normalized",
                "",
            ]
        )
    return "\n".join(values)


class FakeModel:
    def __init__(self) -> None:
        self.snapshots: list[list[dict[str, Any]]] = []

    def query(self, messages):
        self.snapshots.append([dict(message) for message in messages])
        return {
            "content": (
                "```bash\nrg -n 'resolve_model' model.py\n```\n"
                "<context_focus_question>Find resolve_model</context_focus_question>"
            )
        }


class FakeAgent:
    def __init__(self) -> None:
        self.model = FakeModel()
        self.messages = [{"role": "system", "content": "system"}]
        self.raw = _source()

    def add_message(self, role, content, **kwargs):
        self.messages.append({"role": role, "content": content, **kwargs})

    def parse_action(self, response):
        command = re.search(r"```bash\n(.*?)\n```", response["content"], re.DOTALL)
        focus = re.search(
            r"<context_focus_question>(.*?)</context_focus_question>",
            response["content"],
            re.DOTALL,
        )
        return {
            "action": command.group(1),
            "context_focus_question": focus.group(1),
        }

    def query(self):
        response = self.model.query(self.messages)
        self.parse_action(response)
        self.add_message("assistant", **response)
        return response

    def execute_action(self, action):
        del action
        return {"output": self.raw, "returncode": 0}

    def get_observation(self, response):
        output = self.execute_action(self.parse_action(response))
        self.add_message("user", f"Observation: {output['output']}")
        return output

    def run(self, task, **kwargs):
        del task, kwargs
        return "Submitted", "done"


def test_mini_adapter_keeps_canonical_history_full_and_sends_temporary_views() -> None:
    config = AgentContextConfig.from_mapping(
        {
            "hot_observations": 1,
            "track_later_references": False,
            "planner": {"mode": "retention", "target_retention": 0.55},
        }
    )
    patch_agent(FakeAgent, config)
    agent = FakeAgent()

    first_action = agent.query()
    agent.get_observation(first_action)
    first_observation = agent.messages[-1]
    canonical_content = first_observation["content"]

    second_action = agent.query()
    assert agent.model.snapshots[1][-1]["content"] == canonical_content
    agent.get_observation(second_action)
    agent.query()

    assert first_observation["content"] == canonical_content
    assert any(
        "agent_context_view" in str(message.get("content", ""))
        for message in agent.model.snapshots[2]
    )
    assert all(
        "agent_context_stats" not in message and "agent_context_manifest" not in message
        for prompt in agent.model.snapshots
        for message in prompt
    )
