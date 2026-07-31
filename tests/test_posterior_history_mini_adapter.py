from __future__ import annotations

import re
from typing import Any

import pytest

from posterior_history_pruning.mini_adapter.config_adapter import adapt_config
from posterior_history_pruning.mini_adapter.hook import (
    SWE_PRUNER_POSTERIOR_MODE,
    _patch,
    assert_mini_compatible,
)
from posterior_history_pruning.protocol import PosteriorHistoryConfig


def _source() -> str:
    return "\n".join(
        f"def {'resolve_model' if index == 30 else f'helper_{index}'}(config):\n"
        f"    value = config.get('key_{index}')\n"
        "    if value is None:\n"
        f"        raise ValueError('key_{index}')\n"
        "    return value\n"
        for index in range(80)
    )


class FakeModel:
    def __init__(self) -> None:
        self.n_calls = 0
        self.cost = 0.0
        self.snapshots: list[list[dict[str, Any]]] = []

    def query(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        self.n_calls += 1
        self.snapshots.append([dict(message) for message in messages])
        return {
            "content": (
                "```bash\nrg -n 'resolve_model' model.py\n```\n"
                "<context_focus_question>Where is resolve_model validated?</context_focus_question>"
            )
        }


class OfficialForkShape:
    def __init__(self) -> None:
        self.model = FakeModel()
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
        ]
        self.raw = _source()

    def add_message(self, role, content, **kwargs):
        self.messages.append({"role": role, "content": content, **kwargs})

    def parse_action(self, response):
        match = re.search(r"```bash\n(.*?)\n```", response["content"], re.DOTALL)
        if not match:
            raise ValueError("missing action")
        focus = re.search(
            r"<context_focus_question>(.*?)</context_focus_question>",
            response["content"],
            re.DOTALL,
        )
        return {
            "action": match.group(1),
            "context_focus_question": focus.group(1) if focus else None,
        }

    def query(self):
        response = self.model.query(self.messages)
        self.parse_action(response)
        self.add_message("assistant", **response)
        return response

    def execute_action(self, action):
        return {"output": self.raw, "returncode": 0}

    def get_observation(self, response):
        output = self.execute_action(self.parse_action(response))
        self.add_message("user", f"Observation: {output['output']}")
        return output

    def run(self, task, **kwargs):
        del task, kwargs
        return "Submitted", "done"


def _config() -> PosteriorHistoryConfig:
    return PosteriorHistoryConfig(
        hot_observations=1,
        min_input_tokens=0,
        min_savings_tokens=1,
        max_retention_ratio=0.99,
        block_max_lines=8,
        max_output_chars=50000,
    )


def test_official_fork_prompt_boundary_is_delayed_and_canonical_history_stays_full() -> None:
    assert assert_mini_compatible(OfficialForkShape) == SWE_PRUNER_POSTERIOR_MODE
    _patch(OfficialForkShape, _config())
    agent = OfficialForkShape()

    initial = agent.query()
    agent.get_observation(initial)
    first_observation = agent.messages[-1]

    # The first model call after an observation sees its complete output.
    second = agent.query()
    assert agent.model.snapshots[1][-1]["content"] == first_observation["content"]
    agent.get_observation(second)
    second_observation = agent.messages[-1]

    # Once another full observation exists, only the older prompt-view copy is
    # compacted. Canonical trajectory messages keep both original outputs.
    agent.query()
    third_prompt = agent.model.snapshots[2]
    first_prompt_observation = next(
        message
        for message in third_prompt
        if message["role"] == "user" and "posterior_history_compaction" in message["content"]
    )
    assert "def resolve_model" in first_prompt_observation["content"]
    assert second_observation["content"] in [message["content"] for message in third_prompt]
    assert "posterior_history_compaction" not in first_observation["content"]
    assert first_observation["content"].startswith("Observation: def helper_0")
    assert agent.model.n_calls == 3
    assert all(
        "posterior_history_stats" not in message
        for prompt in agent.model.snapshots
        for message in prompt
    )


def test_signature_guard_refuses_changed_query_boundary() -> None:
    class WrongQuery:
        def query(self, messages):
            return messages

        def get_observation(self, response):
            return response

        def run(self, task, **kwargs):
            return task, kwargs

        def execute_action(self, action):
            return action

        def add_message(self, role, content, **kwargs):
            return role, content, kwargs

    with pytest.raises(RuntimeError, match="query signature changed"):
        assert_mini_compatible(WrongQuery)


def test_config_adapter_keeps_prompt_identical_while_removing_legacy_pruner() -> None:
    base = {
        "model": {"model_name": "old", "model_kwargs": {}},
        "agent": {
            "pruner": {"url": "http://legacy"},
            "system_template": "Emit <context_focus_question> before each action.",
        },
    }
    result = adapt_config(base, model_id="Qwen3.5-27B", api_base="http://127.0.0.1:8015/v1")

    assert result["model"]["model_name"] == "hosted_vllm/Qwen3.5-27B"
    assert result["agent"]["system_template"] == base["agent"]["system_template"]
    assert "pruner" not in result["agent"]
    assert "pruner" in base["agent"]
