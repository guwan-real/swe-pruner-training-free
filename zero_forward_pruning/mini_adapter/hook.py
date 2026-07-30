from __future__ import annotations

import inspect
import logging
from pathlib import PurePosixPath
from typing import Any, Mapping

from zero_forward_pruning.mini_adapter.client import (
    ZeroForwardClient,
    ZeroForwardClientConfig,
)
from zero_forward_pruning.text import estimate_tokens, extract_paths, shell_verb

LOGGER = logging.getLogger("zero_forward_pruning.mini_adapter")
_INSTALLED = False
_ORIGINAL_EXECUTE_ACTIONS: Any = None


def assert_mini_compatible(default_agent_class: type) -> None:
    execute_actions = getattr(default_agent_class, "execute_actions", None)
    add_messages = getattr(default_agent_class, "add_messages", None)
    if not callable(execute_actions) or not callable(add_messages):
        raise RuntimeError(
            "mini-swe-agent DefaultAgent must expose execute_actions() and add_messages()"
        )
    signature = inspect.signature(execute_actions)
    if tuple(signature.parameters) != ("self", "message"):
        raise RuntimeError(
            "mini-swe-agent DefaultAgent.execute_actions signature changed; "
            "expected execute_actions(self, message)"
        )


def _stats(result: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "method",
        "status",
        "origin_token_cnt",
        "left_token_cnt",
        "model_input_token_cnt",
        "model_forward_count",
        "llm_token_count",
        "original_line_count",
        "kept_line_count",
        "retention_ratio",
        "latency_ms",
        "raw_id",
        "recovery_url",
        "error",
        "diagnostics",
    )
    return {name: result.get(name) for name in fields if name in result}


def _query_from_action(action: Mapping[str, Any], task: str) -> str:
    explicit = action.get("context_focus_question")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    command = action.get("command", "")
    command = command if isinstance(command, str) else ""
    paths = extract_paths(command)
    # The task is sent separately.  Keeping the query compact avoids spending
    # CPU ranking boilerplate task text that has no overlap with tool output.
    parts = [command]
    if paths:
        parts.append(" ".join(PurePosixPath(path).name for path in paths))
    if not command.strip():
        parts.append(task[-1000:])
    return "\n".join(part for part in parts if part)


def _recent_context(agent: Any) -> str:
    messages = getattr(agent, "messages", [])
    if not isinstance(messages, list):
        return ""
    values: list[str] = []
    for message in messages[-3:]:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if isinstance(content, str):
            values.append(content[-1000:])
    return "\n".join(values)[-2500:]


def apply_to_output(
    agent: Any,
    *,
    action: Mapping[str, Any],
    output: Mapping[str, Any],
    client: ZeroForwardClient,
    action_index: int,
) -> dict[str, Any]:
    result_output = dict(output)
    code = output.get("output", "")
    if not isinstance(code, str):
        return result_output
    command = action.get("command", "")
    command = command if isinstance(command, str) else ""
    if "/raw/" in command and shell_verb(command) in {"curl", "wget"}:
        tokens = estimate_tokens(code)
        line_count = len(code.splitlines())
        extra = dict(result_output.get("extra", {}))
        extra["zero_forward_pruning"] = {
            "method": "recovery_bypass",
            "status": "skipped",
            "origin_token_cnt": tokens,
            "left_token_cnt": tokens,
            "model_input_token_cnt": 0,
            "model_forward_count": 0,
            "llm_token_count": 0,
            "original_line_count": line_count,
            "kept_line_count": line_count,
            "retention_ratio": 1.0,
            "diagnostics": {"reason": "raw-recovery-action-bypass"},
        }
        result_output["extra"] = extra
        return result_output
    extra_vars = getattr(agent, "extra_template_vars", {})
    task = extra_vars.get("task", "") if isinstance(extra_vars, Mapping) else ""
    task = task if isinstance(task, str) else ""
    instance_id = getattr(agent, "instance_id", "") or "unknown"
    call_count = getattr(agent, "n_calls", 0)
    paths = extract_paths(command)
    try:
        result = client.prune(
            code=code,
            query=_query_from_action(action, task),
            command=command,
            path=paths[0] if paths else "",
            task=task,
            recent_context=_recent_context(agent),
            request_id=f"{instance_id}:call-{call_count}:action-{action_index}",
            metadata={
                "adapter": "mini-swe-agent-tool-boundary-v1",
                "returncode": output.get("returncode"),
            },
        )
        result_output["output"] = result["pruned_code"]
        extra = dict(result_output.get("extra", {}))
        extra["zero_forward_pruning"] = _stats(result)
        result_output["extra"] = extra
    except Exception as exc:
        LOGGER.warning("zero-forward pruning failed open: %s", exc)
        extra = dict(result_output.get("extra", {}))
        extra["zero_forward_pruning"] = {
            "method": "client",
            "status": "client_error",
            "model_input_token_cnt": 0,
            "model_forward_count": 0,
            "llm_token_count": 0,
            "error": str(exc),
        }
        result_output["extra"] = extra
    return result_output


def install_hook(config: ZeroForwardClientConfig | None = None) -> bool:
    """Patch only the tool-output boundary of the current mini process."""

    global _INSTALLED, _ORIGINAL_EXECUTE_ACTIONS
    if _INSTALLED:
        return True
    config = config if config is not None else ZeroForwardClientConfig.from_env()
    if config is None:
        return False
    try:
        from minisweagent.agents.default import DefaultAgent
    except ImportError as exc:
        raise RuntimeError(
            "minisweagent is not importable; use its existing Python environment"
        ) from exc
    assert_mini_compatible(DefaultAgent)
    client = ZeroForwardClient(config)
    original_execute_actions = DefaultAgent.execute_actions

    def execute_actions_with_zero_forward(self: Any, message: dict[str, Any]) -> list[dict]:
        if getattr(self, "pruner_client", None) is not None:
            raise RuntimeError(
                "legacy mini-swe-agent pruner is active together with zero-forward "
                "adapter; remove agent.pruner or use --disable-pruner"
            )
        actions = message.get("extra", {}).get("actions", [])
        outputs = []
        for index, action in enumerate(actions):
            raw_output = self.env.execute(action)
            outputs.append(
                apply_to_output(
                    self,
                    action=action,
                    output=raw_output,
                    client=client,
                    action_index=index,
                )
            )
        observations = self.model.format_observation_messages(
            message,
            outputs,
            self.get_template_vars(),
        )
        return self.add_messages(*observations)

    _ORIGINAL_EXECUTE_ACTIONS = original_execute_actions
    DefaultAgent.execute_actions = execute_actions_with_zero_forward
    _INSTALLED = True
    LOGGER.info(
        "installed zero-forward mini-swe-agent tool-boundary hook: %s",
        config.endpoint,
    )
    return True
