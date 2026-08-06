from __future__ import annotations

import inspect
import json
import logging
import os
from functools import wraps
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from agent_context.config import AgentContextConfig
from agent_context.engine import ContextEngine
from agent_context.models import ActionEvent
from posterior_history_pruning.state import locate_observation_boundary

LOGGER = logging.getLogger("agent_context.adapters.mini_swe")
MINI_SWE_AGENT_CONTEXT_MODE = "mini-swe-agent-context-v1"
_INSTALLED = False
_ENGINE_ATTRIBUTE = "_agent_context_engine"


def _signature_names(callable_object: Any) -> tuple[str, ...]:
    return tuple(inspect.signature(callable_object).parameters)


def assert_mini_compatible(default_agent_class: type) -> str:
    required = ("query", "get_observation", "run", "execute_action", "add_message")
    missing = [name for name in required if not callable(getattr(default_agent_class, name, None))]
    if missing:
        raise RuntimeError(
            "unsupported mini-swe-agent DefaultAgent; missing methods: " + ", ".join(missing)
        )
    if _signature_names(default_agent_class.query) != ("self",):
        raise RuntimeError("mini-swe-agent query signature changed; expected query(self)")
    if _signature_names(default_agent_class.get_observation) != ("self", "response"):
        raise RuntimeError(
            "mini-swe-agent get_observation signature changed; expected get_observation(self, response)"
        )
    if _signature_names(default_agent_class.execute_action) != ("self", "action"):
        raise RuntimeError(
            "mini-swe-agent execute_action signature changed; expected execute_action(self, action)"
        )
    add_parameters = tuple(inspect.signature(default_agent_class.add_message).parameters.values())
    if (
        len(add_parameters) != 4
        or tuple(parameter.name for parameter in add_parameters[:3]) != ("self", "role", "content")
        or add_parameters[3].kind is not inspect.Parameter.VAR_KEYWORD
    ):
        raise RuntimeError(
            "mini-swe-agent add_message signature changed; expected add_message(self, role, content, **kwargs)"
        )
    return MINI_SWE_AGENT_CONTEXT_MODE


def _engine(agent: Any, config: AgentContextConfig) -> ContextEngine:
    current = getattr(agent, _ENGINE_ATTRIBUTE, None)
    if isinstance(current, ContextEngine):
        return current
    current = ContextEngine(config)
    current.start_task("mini-task")
    setattr(agent, _ENGINE_ATTRIBUTE, current)
    return current


def _action_path(command: str) -> str:
    suffixes = {
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".go",
        ".rs",
        ".java",
        ".c",
        ".cc",
        ".cpp",
        ".h",
        ".hpp",
        ".rb",
        ".php",
        ".sh",
    }
    for word in command.replace("'", " ").replace('"', " ").split():
        candidate = word.strip("()[]{};,:")
        if PurePosixPath(candidate).suffix.lower() in suffixes:
            return candidate
    return ""


def _action_event(agent: Any, response: Mapping[str, Any], *, step: int) -> ActionEvent:
    action: Mapping[str, Any] = {}
    try:
        parsed = agent.parse_action(dict(response))
        if isinstance(parsed, Mapping):
            action = parsed
    except Exception:  # pragma: no cover - upstream already parsed successful responses
        pass
    command = action.get("action", "")
    focus = action.get("context_focus_question", "")
    content = response.get("content", "")
    return ActionEvent(
        step=step,
        command=command if isinstance(command, str) else "",
        context_focus_question=focus if isinstance(focus, str) else "",
        response_content=content if isinstance(content, str) else "",
    )


def _append_assistant(
    canonical_messages: list[dict[str, Any]],
    prompt_view: list[dict[str, Any]],
    original_length: int,
    manifest: Mapping[str, Any],
) -> None:
    appended = prompt_view[original_length:]
    if len(appended) != 1 or appended[0].get("role") != "assistant":
        raise RuntimeError(
            "mini-swe-agent query lifecycle changed; expected one assistant message append"
        )
    appended[0]["agent_context_manifest"] = dict(manifest)
    canonical_messages.append(appended[0])


def patch_agent(default_agent_class: type, config: AgentContextConfig) -> None:
    assert_mini_compatible(default_agent_class)
    original_query = default_agent_class.query
    original_get_observation = default_agent_class.get_observation
    original_run = default_agent_class.run

    @wraps(original_query)
    def query_with_context(self: Any) -> dict[str, Any]:
        engine = _engine(self, config)
        canonical_messages = self.messages
        build = engine.build_prompt(canonical_messages)
        prompt_view = build.as_list()
        original_length = len(prompt_view)
        self.messages = prompt_view
        response: dict[str, Any] | None = None
        try:
            response = original_query(self)
        finally:
            self.messages = canonical_messages
        if response is None:
            raise RuntimeError("mini-swe-agent query returned no response")
        _append_assistant(
            canonical_messages,
            prompt_view,
            original_length,
            build.manifest.to_dict(),
        )
        engine.observe_action(_action_event(self, response, step=engine.step + 1))
        return response

    @wraps(original_get_observation)
    def get_observation_with_context(self: Any, response: dict[str, Any]) -> dict[str, Any]:
        engine = _engine(self, config)
        original_length = len(self.messages)
        output = original_get_observation(self, response)
        if not isinstance(output, Mapping):
            return output
        raw_output = output.get("output")
        added = self.messages[original_length:]
        message = next(
            (
                item
                for item in reversed(added)
                if isinstance(item, dict) and item.get("role") == "user"
            ),
            None,
        )
        if not isinstance(message, dict) or not isinstance(raw_output, str) or not raw_output:
            return output
        content = message.get("content")
        boundary = (
            locate_observation_boundary(content, raw_output) if isinstance(content, str) else None
        )
        if boundary is None:
            message["agent_context_stats"] = {
                "status": "untracked",
                "reason": "rendered-output-boundary-not-found",
            }
            return output
        causing = _action_event(self, response, step=engine.step)
        observation = engine.ingest_observation(
            message,
            visible_content=boundary.selection_output,
            raw_content=raw_output,
            causing_action=causing.command,
            path=_action_path(causing.command),
            prefix=boundary.prefix,
            suffix=boundary.suffix,
            metadata={"boundary_mode": boundary.mode},
            boundary_is_validated=True,
        )
        message["agent_context_stats"] = {
            "status": "tracked",
            "observation_id": observation.id,
            "kind": observation.kind.value,
            "boundary_mode": boundary.mode,
        }
        return output

    @wraps(original_run)
    def run_with_context(self: Any, task: str, **kwargs: Any):
        engine = _engine(self, config)
        task_id = str(kwargs.get("instance_id") or kwargs.get("task_id") or "mini-task")
        engine.start_task(task_id, task_text=task)
        try:
            return original_run(self, task, **kwargs)
        finally:
            if self.messages:
                self.messages[-1]["agent_context_report"] = engine.finish_task()

    default_agent_class.query = query_with_context
    default_agent_class.get_observation = get_observation_with_context
    default_agent_class.run = run_with_context


def config_from_env() -> AgentContextConfig | None:
    if os.getenv("AGENT_CONTEXT_ENABLED", "0") != "1":
        return None
    config_path = os.getenv("AGENT_CONTEXT_CONFIG")
    if config_path:
        payload = json.loads(Path(config_path).expanduser().read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("AGENT_CONTEXT_CONFIG must contain a JSON object")
        return AgentContextConfig.from_mapping(payload)
    return AgentContextConfig.from_mapping(
        {
            "timing": os.getenv("AGENT_CONTEXT_TIMING", "posterior"),
            "hot_observations": int(os.getenv("AGENT_CONTEXT_HOT_OBSERVATIONS", "2")),
            "planner": {
                "mode": os.getenv("AGENT_CONTEXT_PLANNER_MODE", "retention"),
                "target_retention": float(os.getenv("AGENT_CONTEXT_TARGET_RETENTION", "0.6")),
            },
        }
    )


def install_hook(config: AgentContextConfig | None = None) -> bool:
    global _INSTALLED
    if _INSTALLED:
        return True
    resolved = config or config_from_env()
    if resolved is None:
        return False
    try:
        from minisweagent.agents.default import DefaultAgent
    except ImportError as exc:
        raise RuntimeError("minisweagent is not importable") from exc
    patch_agent(DefaultAgent, resolved)
    _INSTALLED = True
    LOGGER.info(
        "installed agent-context hook timing=%s planner=%s",
        resolved.timing,
        resolved.planner.mode,
    )
    return True
