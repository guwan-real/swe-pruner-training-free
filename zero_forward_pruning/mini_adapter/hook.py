from __future__ import annotations

import inspect
import logging
import re
import shlex
from functools import wraps
from pathlib import PurePosixPath
from typing import Any, Mapping

from zero_forward_pruning.mini_adapter.client import (
    ZeroForwardClient,
    ZeroForwardClientConfig,
)
from zero_forward_pruning.text import estimate_tokens, extract_paths, shell_verb

LOGGER = logging.getLogger("zero_forward_pruning.mini_adapter")
UPSTREAM_BATCH_MODE = "upstream-batch-v2"
SWE_PRUNER_SINGLE_MODE = "swe-pruner-single-v1"
_INSTALLED = False
_RECOVERY_PATHS_ATTRIBUTE = "_zero_forward_recovery_paths"
_RECOVERY_URL_PATTERN = re.compile(r"https?://[^\s'\";|]+/raw/[A-Za-z0-9_-]+")


def assert_mini_compatible(default_agent_class: type) -> str:
    """Return the supported runtime shape instead of trusting a package version."""

    execute_actions = getattr(default_agent_class, "execute_actions", None)
    add_messages = getattr(default_agent_class, "add_messages", None)
    execute_action = getattr(default_agent_class, "execute_action", None)
    add_message = getattr(default_agent_class, "add_message", None)
    has_batch_api = callable(execute_actions) and callable(add_messages)
    has_single_api = callable(execute_action) and callable(add_message)
    if has_batch_api and has_single_api:
        raise RuntimeError(
            "mini-swe-agent DefaultAgent exposes both supported action APIs; "
            "refusing an ambiguous hook target"
        )
    if has_batch_api:
        signature = inspect.signature(execute_actions)
        if tuple(signature.parameters) != ("self", "message"):
            raise RuntimeError(
                "mini-swe-agent DefaultAgent.execute_actions signature changed; "
                "expected execute_actions(self, message)"
            )
        add_signature = inspect.signature(add_messages)
        add_parameters = tuple(add_signature.parameters.values())
        if (
            len(add_parameters) != 2
            or add_parameters[0].name != "self"
            or add_parameters[1].name != "messages"
            or add_parameters[1].kind is not inspect.Parameter.VAR_POSITIONAL
        ):
            raise RuntimeError(
                "mini-swe-agent DefaultAgent.add_messages signature changed; "
                "expected add_messages(self, *messages)"
            )
        return UPSTREAM_BATCH_MODE
    if has_single_api:
        signature = inspect.signature(execute_action)
        if tuple(signature.parameters) != ("self", "action"):
            raise RuntimeError(
                "SWE-Pruner mini fork DefaultAgent.execute_action signature changed; "
                "expected execute_action(self, action)"
            )
        add_signature = inspect.signature(add_message)
        add_parameters = tuple(add_signature.parameters.values())
        if (
            len(add_parameters) != 4
            or tuple(parameter.name for parameter in add_parameters[:3])
            != ("self", "role", "content")
            or add_parameters[3].kind is not inspect.Parameter.VAR_KEYWORD
        ):
            raise RuntimeError(
                "SWE-Pruner mini fork DefaultAgent.add_message signature changed; "
                "expected add_message(self, role, content, **kwargs)"
            )
        return SWE_PRUNER_SINGLE_MODE
    available = sorted(
        name
        for name in ("execute_actions", "add_messages", "execute_action", "add_message")
        if callable(getattr(default_agent_class, name, None))
    )
    raise RuntimeError(
        "unsupported mini-swe-agent DefaultAgent action API; expected "
        "execute_actions(self, message) + add_messages() or the official SWE-Pruner "
        "eval fork's execute_action(self, action) + add_message(); "
        f"found callable methods: {available}"
    )


def detect_installed_mini_mode() -> str:
    try:
        from minisweagent.agents.default import DefaultAgent
    except ImportError as exc:
        raise RuntimeError(
            "minisweagent is not importable; use its existing Python environment"
        ) from exc
    return assert_mini_compatible(DefaultAgent)


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


def _normalize_action(action: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(action)
    command = normalized.get("command")
    if not isinstance(command, str):
        fork_action = normalized.get("action")
        normalized["command"] = fork_action if isinstance(fork_action, str) else ""
    return normalized


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


def _agent_call_count(agent: Any) -> int:
    candidates = (
        getattr(agent, "n_calls", None),
        getattr(getattr(agent, "model", None), "n_calls", None),
    )
    for value in candidates:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return 0


def _zero_forward_stats(output: Mapping[str, Any]) -> dict[str, Any] | None:
    extra = output.get("extra")
    if not isinstance(extra, Mapping):
        return None
    stats = extra.get("zero_forward_pruning")
    return dict(stats) if isinstance(stats, Mapping) else None


def _copy_extra(output: Mapping[str, Any]) -> dict[str, Any]:
    extra = output.get("extra")
    return dict(extra) if isinstance(extra, Mapping) else {}


def _recovery_output_path(command: str) -> str:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return ""
    verb = shell_verb(command)
    if verb not in {"curl", "wget"}:
        return ""
    short_option = "-o" if verb == "curl" else "-O"
    long_option = "--output" if verb == "curl" else "--output-document"
    for index, token in enumerate(tokens):
        if token in {short_option, long_option}:
            if index + 1 < len(tokens):
                output_path = tokens[index + 1]
                return "" if output_path in {"-", "/dev/stdout", "/dev/fd/1"} else output_path
        if token.startswith(f"{long_option}="):
            output_path = token.removeprefix(f"{long_option}=")
            return "" if output_path in {"-", "/dev/stdout", "/dev/fd/1"} else output_path
        if token.startswith(short_option) and len(token) > len(short_option):
            output_path = token.removeprefix(short_option)
            return "" if output_path in {"-", "/dev/stdout", "/dev/fd/1"} else output_path
    return ""


def _remember_recovery_path(agent: Any, path: str) -> None:
    if not path:
        return
    known = getattr(agent, _RECOVERY_PATHS_ATTRIBUTE, None)
    if not isinstance(known, set):
        known = set()
        setattr(agent, _RECOVERY_PATHS_ATTRIBUTE, known)
    known.add(path)


def _known_recovery_path(agent: Any, command: str) -> str:
    known = getattr(agent, _RECOVERY_PATHS_ATTRIBUTE, set())
    if not isinstance(known, set):
        return ""
    return next(
        (
            path
            for path in sorted(known, key=len, reverse=True)
            if isinstance(path, str) and path and path in command
        ),
        "",
    )


def _recovery_url(command: str) -> str:
    match = _RECOVERY_URL_PATTERN.search(command)
    return match.group(0) if match else ""


def _recovery_guard_message(
    *,
    code: str,
    saved_path: str,
    recovery_url: str,
) -> str:
    line_count = len(code.splitlines())
    token_count = estimate_tokens(code)
    if saved_path:
        quoted_path = shlex.quote(saved_path)
        location = f"The exact output is saved at {quoted_path}."
        fetch = ""
    else:
        fallback_path = "/tmp/zero-forward-recovered.txt"
        quoted_path = shlex.quote(fallback_path)
        location = "The exact output remains available from the recovery URL."
        fetch = (
            f"\nSave it without printing it:\n"
            f"curl -fsS {shlex.quote(recovery_url)} -o {quoted_path}"
            if recovery_url
            else ""
        )
    return (
        "<zero_forward_recovery_guard>\n"
        f"Withheld an unbounded recovery echo ({line_count} lines, ~{token_count} tokens) "
        "so it does not remain in every later model prompt.\n"
        f"{location}{fetch}\n"
        "Inspect only the needed evidence with bounded commands, for example:\n"
        f"rg -n 'SYMBOL|ERROR|TODO' {quoted_path}\n"
        f"sed -n '1,60p' {quoted_path}  # adjust the roughly 40-60 line range\n"
        f"Do not use cat {quoted_path} or otherwise print the complete file.\n"
        "</zero_forward_recovery_guard>"
    )


def _handle_recovery_output(
    agent: Any,
    *,
    command: str,
    code: str,
    output: Mapping[str, Any],
    max_chars: int,
) -> dict[str, Any]:
    result_output = dict(output)
    output_path = _recovery_output_path(command)
    if output_path:
        _remember_recovery_path(agent, output_path)
    known_path = output_path or _known_recovery_path(agent, command)
    origin_tokens = estimate_tokens(code)
    line_count = len(code.splitlines())
    extra = _copy_extra(result_output)
    returncode = output.get("returncode")
    command_succeeded = returncode is None or returncode == 0
    if len(code) > max_chars and command_succeeded:
        guarded = _recovery_guard_message(
            code=code,
            saved_path=known_path,
            recovery_url=_recovery_url(command),
        )
        result_output["output"] = guarded
        left_tokens = estimate_tokens(guarded)
        extra["zero_forward_pruning"] = {
            "method": "recovery_guard",
            "status": "guarded",
            "origin_token_cnt": origin_tokens,
            "left_token_cnt": left_tokens,
            "model_input_token_cnt": 0,
            "model_forward_count": 0,
            "llm_token_count": 0,
            "original_line_count": line_count,
            "kept_line_count": len(guarded.splitlines()),
            "retention_ratio": left_tokens / origin_tokens if origin_tokens else 1.0,
            "diagnostics": {
                "reason": "unbounded-recovery-output-withheld",
                "recovery_output_max_chars": max_chars,
                "saved_path": known_path,
                "recovery_url": _recovery_url(command),
            },
        }
    else:
        reason = (
            "bounded-recovery-output-bypass"
            if command_succeeded
            else "recovery-command-failed-bypass"
        )
        extra["zero_forward_pruning"] = {
            "method": "recovery_bypass",
            "status": "skipped",
            "origin_token_cnt": origin_tokens,
            "left_token_cnt": origin_tokens,
            "model_input_token_cnt": 0,
            "model_forward_count": 0,
            "llm_token_count": 0,
            "original_line_count": line_count,
            "kept_line_count": line_count,
            "retention_ratio": 1.0,
            "diagnostics": {"reason": reason},
        }
    result_output["extra"] = extra
    return result_output


def _publish_swe_pruner_stats(output: dict[str, Any]) -> dict[str, Any]:
    """Expose stats where the official SWE-Pruner fork stores trajectory metadata."""

    stats = _zero_forward_stats(output)
    if stats is not None:
        output["pruned_stats"] = stats
    return output


def apply_to_output(
    agent: Any,
    *,
    action: Mapping[str, Any],
    output: Mapping[str, Any],
    client: ZeroForwardClient,
    action_index: int,
) -> dict[str, Any]:
    action = _normalize_action(action)
    result_output = dict(output)
    code = output.get("output", "")
    if not isinstance(code, str):
        return result_output
    command = action.get("command", "")
    command = command if isinstance(command, str) else ""
    is_recovery_fetch = "/raw/" in command and shell_verb(command) in {"curl", "wget"}
    is_known_recovery_read = bool(_known_recovery_path(agent, command))
    if is_recovery_fetch or is_known_recovery_read:
        config = getattr(client, "config", None)
        recovery_max_chars = getattr(config, "recovery_max_chars", 3000)
        return _handle_recovery_output(
            agent,
            command=command,
            code=code,
            output=result_output,
            max_chars=recovery_max_chars,
        )
    extra_vars = getattr(agent, "extra_template_vars", {})
    task = extra_vars.get("task", "") if isinstance(extra_vars, Mapping) else ""
    task = task if isinstance(task, str) else ""
    instance_id = getattr(agent, "instance_id", "") or "unknown"
    call_count = _agent_call_count(agent)
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
                "adapter": "mini-swe-agent-tool-boundary-v2",
                "returncode": output.get("returncode"),
            },
        )
        result_output["output"] = result["pruned_code"]
        extra = _copy_extra(result_output)
        extra["zero_forward_pruning"] = _stats(result)
        result_output["extra"] = extra
    except Exception as exc:
        LOGGER.warning("zero-forward pruning failed open: %s", exc)
        extra = _copy_extra(result_output)
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


def _reject_active_legacy_pruner(agent: Any) -> None:
    if getattr(agent, "pruner_client", None) is not None:
        raise RuntimeError(
            "legacy SWE-Pruner PrunerClient is active together with the zero-forward "
            "adapter; remove agent.pruner from the generated config"
        )


def _patch_upstream_batch(default_agent_class: type, client: ZeroForwardClient) -> None:
    original_execute_actions = default_agent_class.execute_actions

    @wraps(original_execute_actions)
    def execute_actions_with_zero_forward(self: Any, message: dict[str, Any]) -> list[dict]:
        _reject_active_legacy_pruner(self)
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

    default_agent_class.execute_actions = execute_actions_with_zero_forward


def _patch_swe_pruner_single(default_agent_class: type, client: ZeroForwardClient) -> None:
    """Wrap the official eval fork without copying its execution/termination flow."""

    original_execute_action = default_agent_class.execute_action

    @wraps(original_execute_action)
    def execute_action_with_zero_forward(
        self: Any,
        action: dict[str, Any],
    ) -> dict[str, Any]:
        # Check before the original call: its final step invokes _apply_pruner.
        _reject_active_legacy_pruner(self)
        raw_output = original_execute_action(self, action)
        compacted = apply_to_output(
            self,
            action=action,
            output=raw_output,
            client=client,
            action_index=0,
        )
        return _publish_swe_pruner_stats(compacted)

    default_agent_class.execute_action = execute_action_with_zero_forward


def install_hook(config: ZeroForwardClientConfig | None = None) -> bool:
    """Patch only the tool-output boundary of the current mini process."""

    global _INSTALLED
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

    mode = assert_mini_compatible(DefaultAgent)
    client = ZeroForwardClient(config)
    if mode == SWE_PRUNER_SINGLE_MODE:
        _patch_swe_pruner_single(DefaultAgent, client)
    else:
        _patch_upstream_batch(DefaultAgent, client)
    _INSTALLED = True
    LOGGER.info(
        "installed zero-forward mini-swe-agent hook mode=%s endpoint=%s",
        mode,
        config.endpoint,
    )
    return True
