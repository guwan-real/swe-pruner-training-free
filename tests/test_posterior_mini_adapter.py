from __future__ import annotations

from types import SimpleNamespace

from posterior_pruning.mini_adapter.config_adapter import adapt_config
from posterior_pruning.mini_adapter.hook import apply_after_query


class FakeClient:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls = []

    def prune(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("service down")
        return {
            "method": "single_verify",
            "status": "accepted",
            "pruned_response": "Observation:\nkept",
            "retention_ratio": 0.5,
            "model_forward_count": 2,
            "candidates_evaluated": 1,
        }


def agent() -> SimpleNamespace:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "first action"},
        {"role": "user", "content": "Observation:\nfull\nresponse"},
        {"role": "assistant", "content": "second action"},
    ]
    return SimpleNamespace(
        messages=messages,
        extra_template_vars={"task": "fix parser"},
        instance_id="repo__issue-1",
        model=SimpleNamespace(n_calls=2),
    )


def test_hook_compacts_previous_observation_only_after_action_exists() -> None:
    value = agent()
    client = FakeClient()

    apply_after_query(value, {"content": "second action"}, client)

    assert client.calls[0]["messages"][3]["content"] == "Observation:\nfull\nresponse"
    assert client.calls[0]["next_action"] == "second action"
    assert value.messages[3]["content"] == "Observation:\nkept"
    assert value.messages[4]["content"] == "second action"
    assert value.messages[3]["posterior_pruned_stats"]["model_forward_count"] == 2


def test_hook_fails_open_and_records_the_error() -> None:
    value = agent()

    apply_after_query(value, {"content": "second action"}, FakeClient(fail=True))

    assert value.messages[3]["content"] == "Observation:\nfull\nresponse"
    stats = value.messages[3]["posterior_pruned_stats"]
    assert stats["status"] == "client_error"
    assert "service down" in stats["error"]
    assert stats["kept_estimated_tokens"] == stats["original_estimated_tokens"]


def test_config_adapter_removes_legacy_pruner_for_fair_arms() -> None:
    base = {
        "model": {"model_name": "old", "model_kwargs": {}},
        "agent": {"pruner": {"url": "http://legacy"}, "system_template": "same prompt"},
    }

    config = adapt_config(
        base,
        model_id="Qwen3.5-27B",
        api_base="http://127.0.0.1:8015/v1",
    )

    assert config["model"]["model_name"] == "hosted_vllm/Qwen3.5-27B"
    assert config["model"]["model_kwargs"]["temperature"] == 0.0
    assert "pruner" not in config["agent"]
    assert config["agent"]["system_template"] == "same prompt"
    assert "pruner" in base["agent"]
