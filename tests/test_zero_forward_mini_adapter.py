from __future__ import annotations

from typing import Any

import pytest

from zero_forward_pruning.mini_adapter.config_adapter import adapt_config
from zero_forward_pruning.mini_adapter.hook import (
    apply_to_output,
    assert_mini_compatible,
)


class FakeClient:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.payload: dict[str, Any] | None = None

    def prune(self, **kwargs):
        self.payload = kwargs
        if self.fail:
            raise RuntimeError("service down")
        return {
            "pruned_code": "COMPACT",
            "method": "adaptive_evidence",
            "status": "pruned",
            "model_input_token_cnt": 0,
            "model_forward_count": 0,
            "llm_token_count": 0,
            "origin_token_cnt": 100,
            "left_token_cnt": 10,
            "raw_id": "raw",
        }


class FakeAgent:
    extra_template_vars = {"task": "Fix resolve_model"}
    instance_id = "instance"
    n_calls = 3
    messages = [{"role": "assistant", "content": "Inspect model.py"}]


def test_tool_output_is_compacted_before_formatting() -> None:
    client = FakeClient()
    result = apply_to_output(
        FakeAgent(),
        action={"command": "sed -n '1,200p' model.py"},
        output={"output": "RAW", "returncode": 0},
        client=client,  # type: ignore[arg-type]
        action_index=0,
    )
    assert result["output"] == "COMPACT"
    assert result["extra"]["zero_forward_pruning"]["model_forward_count"] == 0
    assert client.payload is not None
    assert client.payload["code"] == "RAW"
    assert client.payload["path"] == "model.py"
    assert client.payload["task"] == "Fix resolve_model"


def test_adapter_fails_open_on_client_error() -> None:
    result = apply_to_output(
        FakeAgent(),
        action={"command": "cat model.py"},
        output={"output": "RAW", "returncode": 0},
        client=FakeClient(fail=True),  # type: ignore[arg-type]
        action_index=0,
    )
    assert result["output"] == "RAW"
    stats = result["extra"]["zero_forward_pruning"]
    assert stats["status"] == "client_error"
    assert stats["model_forward_count"] == 0


def test_raw_recovery_action_is_not_pruned_again() -> None:
    client = FakeClient()
    raw = "full recovered observation\n" * 100
    result = apply_to_output(
        FakeAgent(),
        action={
            "command": (
                "curl -fsS http://host.docker.internal:8124/raw/unguessable-recovery-identifier"
            )
        },
        output={"output": raw, "returncode": 0},
        client=client,  # type: ignore[arg-type]
        action_index=0,
    )
    assert result["output"] == raw
    assert client.payload is None
    stats = result["extra"]["zero_forward_pruning"]
    assert stats["method"] == "recovery_bypass"
    assert stats["status"] == "skipped"
    assert stats["diagnostics"]["reason"] == "raw-recovery-action-bypass"
    assert stats["model_forward_count"] == 0


def test_mini_signature_guard() -> None:
    class Compatible:
        def execute_actions(self, message):
            return []

        def add_messages(self, *messages):
            return list(messages)

    class Incompatible:
        def execute_actions(self, message, extra):
            return []

        def add_messages(self, *messages):
            return list(messages)

    assert_mini_compatible(Compatible)
    with pytest.raises(RuntimeError, match="signature changed"):
        assert_mini_compatible(Incompatible)


def test_config_adapter_removes_legacy_pruner_and_adds_recovery_host() -> None:
    base = {
        "model": {"model_name": "old", "model_kwargs": {}},
        "agent": {"pruner": {"url": "http://old"}},
        "environment": {"run_args": ["--rm"]},
    }
    result = adapt_config(
        base,
        model_id="Qwen3.5-27B",
        api_base="http://127.0.0.1:8015/v1",
    )
    assert result["model"]["model_name"] == "hosted_vllm/Qwen3.5-27B"
    assert "pruner" not in result["agent"]
    assert "--add-host=host.docker.internal:host-gateway" in result["environment"]["run_args"]
    assert "pruner" in base["agent"]
