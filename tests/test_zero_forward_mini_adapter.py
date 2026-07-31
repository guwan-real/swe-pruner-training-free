from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from zero_forward_pruning.mini_adapter.config_adapter import adapt_config
from zero_forward_pruning.mini_adapter.hook import (
    SWE_PRUNER_SINGLE_MODE,
    UPSTREAM_BATCH_MODE,
    _patch_swe_pruner_single,
    apply_to_output,
    assert_mini_compatible,
)
from zero_forward_pruning.mini_adapter.swebench import _configure_legacy_pruner


class FakeClient:
    def __init__(self, *, fail: bool = False, recovery_max_chars: int = 3000):
        self.fail = fail
        self.payload: dict[str, Any] | None = None
        self.config = SimpleNamespace(recovery_max_chars=recovery_max_chars)

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
    assert stats["diagnostics"]["reason"] == "bounded-recovery-output-bypass"
    assert stats["model_forward_count"] == 0


def test_failed_recovery_output_is_not_hidden_even_when_large() -> None:
    client = FakeClient(recovery_max_chars=256)
    raw = "curl: transfer failed\n" * 100
    result = apply_to_output(
        FakeAgent(),
        action={"command": "curl -fsS http://host.docker.internal:8124/raw/random-id"},
        output={"output": raw, "returncode": 22},
        client=client,  # type: ignore[arg-type]
        action_index=0,
    )

    assert result["output"] == raw
    assert client.payload is None
    assert (
        result["extra"]["zero_forward_pruning"]["diagnostics"]["reason"]
        == "recovery-command-failed-bypass"
    )


def test_stdout_recovery_target_is_not_reported_as_saved_file() -> None:
    client = FakeClient(recovery_max_chars=256)
    result = apply_to_output(
        FakeAgent(),
        action={"command": ("curl -fsS http://host.docker.internal:8124/raw/random-id -o -")},
        output={"output": "x" * 400, "returncode": 0},
        client=client,  # type: ignore[arg-type]
        action_index=0,
    )

    assert result["extra"]["zero_forward_pruning"]["status"] == "guarded"
    assert "remains available from the recovery URL" in result["output"]
    assert "saved at -" not in result["output"]
    assert client.payload is None


def test_large_recovery_echo_is_withheld_but_saved_file_remains_available() -> None:
    client = FakeClient(recovery_max_chars=1000)
    agent = FakeAgent()
    raw = "class SeparableModel:\n    value = 1\n" * 400
    result = apply_to_output(
        agent,
        action={
            "command": (
                "curl -fsS 'http://host.docker.internal:8124/raw/random-id' "
                "-o /tmp/separable_full.py && cat /tmp/separable_full.py"
            )
        },
        output={"output": raw, "returncode": 0},
        client=client,  # type: ignore[arg-type]
        action_index=0,
    )

    assert result["output"] != raw
    assert "zero_forward_recovery_guard" in result["output"]
    assert "/tmp/separable_full.py" in result["output"]
    assert "Do not use cat" in result["output"]
    assert client.payload is None
    stats = result["extra"]["zero_forward_pruning"]
    assert stats["method"] == "recovery_guard"
    assert stats["status"] == "guarded"
    assert stats["left_token_cnt"] < stats["origin_token_cnt"]
    assert stats["diagnostics"]["saved_path"] == "/tmp/separable_full.py"

    # A later unbounded read of the remembered recovery file is guarded too,
    # even though the raw URL no longer appears in the command.
    repeated = apply_to_output(
        agent,
        action={"command": "cat /tmp/separable_full.py"},
        output={"output": raw, "returncode": 0},
        client=client,  # type: ignore[arg-type]
        action_index=0,
    )
    assert repeated["extra"]["zero_forward_pruning"]["status"] == "guarded"
    assert repeated["output"] != raw


def test_bounded_recovery_slice_is_returned_unchanged() -> None:
    client = FakeClient(recovery_max_chars=1000)
    agent = FakeAgent()
    first = apply_to_output(
        agent,
        action={
            "command": (
                "curl -fsS 'http://host.docker.internal:8124/raw/random-id' -o /tmp/recovered.py"
            )
        },
        output={"output": "", "returncode": 0},
        client=client,  # type: ignore[arg-type]
        action_index=0,
    )
    assert first["output"] == ""
    bounded = "def relevant_function():\n    return True\n"
    second = apply_to_output(
        agent,
        action={"command": "sed -n '40,60p' /tmp/recovered.py"},
        output={"output": bounded, "returncode": 0},
        client=client,  # type: ignore[arg-type]
        action_index=0,
    )
    assert second["output"] == bounded
    assert second["extra"]["zero_forward_pruning"]["method"] == "recovery_bypass"


def test_mini_signature_guard() -> None:
    class UpstreamCompatible:
        def execute_actions(self, message):
            return []

        def add_messages(self, *messages):
            return list(messages)

    class SwePrunerCompatible:
        def execute_action(self, action):
            return action

        def add_message(self, role, content, **kwargs):
            return None

    class Incompatible:
        def execute_actions(self, message, extra):
            return []

        def add_messages(self, *messages):
            return list(messages)

    class IncompatibleForkMessages:
        def execute_action(self, action):
            return action

        def add_message(self, role, content):
            return None

    assert assert_mini_compatible(UpstreamCompatible) == UPSTREAM_BATCH_MODE
    assert assert_mini_compatible(SwePrunerCompatible) == SWE_PRUNER_SINGLE_MODE
    with pytest.raises(RuntimeError, match="signature changed"):
        assert_mini_compatible(Incompatible)
    with pytest.raises(RuntimeError, match="add_message signature changed"):
        assert_mini_compatible(IncompatibleForkMessages)


def test_official_swe_pruner_single_action_flow_is_wrapped_once() -> None:
    class Model:
        n_calls = 7

    class Environment:
        calls: list[str]

        def __init__(self):
            self.calls = []

        def execute(self, command):
            self.calls.append(command)
            return {"output": "RAW", "returncode": 0}

    class SwePrunerForkAgent:
        def __init__(self):
            self.env = Environment()
            self.model = Model()
            self.messages = [{"role": "assistant", "content": "inspect the resolver"}]
            self.extra_template_vars = {"task": "Fix resolver"}
            self.instance_id = "astropy__astropy-1"
            self.pruner_client = None
            self.legacy_apply_calls = 0
            self.finished_checks = 0

        def add_message(self, role, content, **kwargs):
            self.messages.append({"role": role, "content": content, **kwargs})

        def has_finished(self, output):
            self.finished_checks += 1

        def _apply_pruner(self, action, output):
            self.legacy_apply_calls += 1
            if self.pruner_client is not None:
                output["output"] = "LEGACY"

        def execute_action(self, action):
            output = self.env.execute(action["action"])
            self.has_finished(output)
            self._apply_pruner(action, output)
            return output

    client = FakeClient()
    _patch_swe_pruner_single(SwePrunerForkAgent, client)  # type: ignore[arg-type]
    agent = SwePrunerForkAgent()
    result = agent.execute_action(
        {
            "action": "sed -n '1,200p' resolver.py",
            "context_focus_question": "Where does resolve_model validate its input?",
        }
    )

    assert agent.env.calls == ["sed -n '1,200p' resolver.py"]
    assert agent.finished_checks == 1
    assert agent.legacy_apply_calls == 1
    assert result["output"] == "COMPACT"
    assert result["pruned_stats"] == result["extra"]["zero_forward_pruning"]
    assert result["pruned_stats"]["model_forward_count"] == 0
    assert client.payload is not None
    assert client.payload["command"] == "sed -n '1,200p' resolver.py"
    assert client.payload["path"] == "resolver.py"
    assert client.payload["query"] == "Where does resolve_model validate its input?"
    assert client.payload["request_id"] == "astropy__astropy-1:call-7:action-0"


def test_official_swe_pruner_hook_rejects_legacy_client_before_execution() -> None:
    class SwePrunerForkAgent:
        def __init__(self):
            self.pruner_client = object()
            self.executed = False

        def add_message(self, role, content, **kwargs):
            return None

        def execute_action(self, action):
            self.executed = True
            return {"output": "LEGACY", "returncode": 0}

    _patch_swe_pruner_single(SwePrunerForkAgent, FakeClient())  # type: ignore[arg-type]
    agent = SwePrunerForkAgent()
    with pytest.raises(RuntimeError, match="legacy SWE-Pruner PrunerClient"):
        agent.execute_action({"action": "cat resolver.py"})
    assert not agent.executed


def test_official_swe_pruner_submission_exception_is_not_swallowed() -> None:
    class Submitted(Exception):
        pass

    class SwePrunerForkAgent:
        pruner_client = None

        def add_message(self, role, content, **kwargs):
            return None

        def execute_action(self, action):
            raise Submitted("patch submitted")

    client = FakeClient()
    _patch_swe_pruner_single(SwePrunerForkAgent, client)  # type: ignore[arg-type]
    with pytest.raises(Submitted, match="patch submitted"):
        SwePrunerForkAgent().execute_action({"action": "submit"})
    assert client.payload is None


def test_swe_pruner_runner_preserves_focus_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    def main(pruner_url=None, disable_pruner=False):
        return None

    monkeypatch.setattr(
        "zero_forward_pruning.mini_adapter.swebench.sys.argv",
        ["runner", "--config", "agent.yaml"],
    )
    _configure_legacy_pruner(main, SWE_PRUNER_SINGLE_MODE)
    from zero_forward_pruning.mini_adapter import swebench

    assert "--disable-pruner" not in swebench.sys.argv


def test_swe_pruner_runner_rejects_flags_that_enable_or_remove_focus_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def main(pruner_url=None, disable_pruner=False):
        return None

    for forbidden, message in (
        ("--pruner-url=http://legacy", "legacy hook"),
        ("--disable-pruner", "context_focus_question"),
    ):
        monkeypatch.setattr(
            "zero_forward_pruning.mini_adapter.swebench.sys.argv",
            ["runner", forbidden],
        )
        with pytest.raises(SystemExit, match=message):
            _configure_legacy_pruner(main, SWE_PRUNER_SINGLE_MODE)


def test_config_adapter_removes_legacy_pruner_and_adds_recovery_host() -> None:
    base = {
        "model": {"model_name": "old", "model_kwargs": {}},
        "agent": {
            "pruner": {"url": "http://old"},
            "system_template": "emit <context_focus_question>...</context_focus_question>",
        },
        "environment": {"run_args": ["--rm"]},
    }
    result = adapt_config(
        base,
        model_id="Qwen3.5-27B",
        api_base="http://127.0.0.1:8015/v1",
    )
    assert result["model"]["model_name"] == "hosted_vllm/Qwen3.5-27B"
    assert "pruner" not in result["agent"]
    assert "context_focus_question" in result["agent"]["system_template"]
    # This is what the official fork's runner does without --disable-pruner.
    # The empty value is false, so DefaultAgent does not construct PrunerClient.
    assert not result["agent"].setdefault("pruner", {})
    assert "--add-host=host.docker.internal:host-gateway" in result["environment"]["run_args"]
    assert "pruner" in base["agent"]
