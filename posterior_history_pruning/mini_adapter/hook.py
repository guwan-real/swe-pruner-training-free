from __future__ import annotations

import inspect
import logging
from functools import wraps
from pathlib import PurePosixPath
from typing import Any, Mapping

from posterior_history_pruning.protocol import PosteriorHistoryConfig, PosteriorSignal
from posterior_history_pruning.state import PosteriorHistoryState

LOGGER = logging.getLogger("posterior_history_pruning.mini_adapter")
SWE_PRUNER_POSTERIOR_MODE = "swe-pruner-posterior-history-v1"
_INSTALLED = False
_STATE_ATTRIBUTE = "_posterior_history_pruning_state"


def _signature_names(callable_object: Any) -> tuple[str, ...]:
    return tuple(inspect.signature(callable_object).parameters)


def assert_mini_compatible(default_agent_class: type) -> str:
    """Verify the exact official SWE-Pruner fork boundary we patch.

    The patch must be applied around ``DefaultAgent.query`` rather than the
    tool executor.  This keeps the freshly produced observation intact for the
    model's first following action.
    """

    required = ("query", "get_observation", "run", "execute_action", "add_message")
    missing = [name for name in required if not callable(getattr(default_agent_class, name, None))]
    if missing:
        raise RuntimeError(
            "unsupported mini-swe-agent DefaultAgent; missing methods: " + ", ".join(missing)
        )
    if _signature_names(default_agent_class.query) != ("self",):
        raise RuntimeError(
            "SWE-Pruner mini fork DefaultAgent.query signature changed; expected query(self)"
        )
    if _signature_names(default_agent_class.get_observation) != ("self", "response"):
        raise RuntimeError(
            "SWE-Pruner mini fork DefaultAgent.get_observation signature changed; "
            "expected get_observation(self, response)"
        )
    if _signature_names(default_agent_class.execute_action) != ("self", "action"):
        raise RuntimeError(
            "SWE-Pruner mini fork DefaultAgent.execute_action signature changed; "
            "expected execute_action(self, action)"
        )
    add_parameters = tuple(inspect.signature(default_agent_class.add_message).parameters.values())
    if (
        len(add_parameters) != 4
        or tuple(parameter.name for parameter in add_parameters[:3]) != ("self", "role", "content")
        or add_parameters[3].kind is not inspect.Parameter.VAR_KEYWORD
    ):
        raise RuntimeError(
            "SWE-Pruner mini fork DefaultAgent.add_message signature changed; "
            "expected add_message(self, role, content, **kwargs)"
        )
    return SWE_PRUNER_POSTERIOR_MODE


def detect_installed_mini_mode() -> str:
    try:
        from minisweagent.agents.default import DefaultAgent
    except ImportError as exc:
        raise RuntimeError(
            "minisweagent is not importable; use the existing SWE-Pruner mini environment"
        ) from exc
    return assert_mini_compatible(DefaultAgent)


def _state(agent: Any, config: PosteriorHistoryConfig) -> PosteriorHistoryState:
    current = getattr(agent, _STATE_ATTRIBUTE, None)
    if isinstance(current, PosteriorHistoryState):
        return current
    current = PosteriorHistoryState(config=config)
    setattr(agent, _STATE_ATTRIBUTE, current)
    return current


def _action_path(command: str) -> str:
    # The selector only needs a conservative source-file hint.  The command
    # itself remains the primary posterior evidence.
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


def _signal_from_response(agent: Any, response: Mapping[str, Any]) -> PosteriorSignal:
    action: Mapping[str, Any] = {}
    try:
        parsed = agent.parse_action(dict(response))
        if isinstance(parsed, Mapping):
            action = parsed
    except Exception:  # pragma: no cover - original query already parsed successfully
        pass
    command = action.get("action", "")
    focus = action.get("context_focus_question", "")
    content = response.get("content", "")
    return PosteriorSignal(
        command=command if isinstance(command, str) else "",
        context_focus_question=focus if isinstance(focus, str) else "",
        response_content=content if isinstance(content, str) else "",
    )


def _causing_action(agent: Any, response: Mapping[str, Any]) -> tuple[str, str]:
    signal = _signal_from_response(agent, response)
    return signal.command, _action_path(signal.command)


def _append_assistant_to_canonical(
    canonical_messages: list[dict[str, Any]],
    prompt_view: list[dict[str, Any]],
    original_length: int,
) -> None:
    appended = prompt_view[original_length:]
    if len(appended) != 1 or appended[0].get("role") != "assistant":
        raise RuntimeError(
            "SWE-Pruner mini fork query lifecycle changed; expected exactly one assistant "
            "message to be appended to the temporary prompt view"
        )
    canonical_messages.append(appended[0])


def _patch(default_agent_class: type, config: PosteriorHistoryConfig) -> None:
    original_query = default_agent_class.query
    original_get_observation = default_agent_class.get_observation
    original_run = default_agent_class.run

    @wraps(original_query)
    def query_with_posterior_history(self: Any) -> dict[str, Any]:
        state = _state(self, config)
        canonical_messages = self.messages
        prompt_view = state.render(canonical_messages)
        original_length = len(prompt_view)
        self.messages = prompt_view
        response: dict[str, Any] | None = None
        try:
            response = original_query(self)
        finally:
            self.messages = canonical_messages
        if response is None:  # pragma: no cover - defensive for a changed upstream method
            raise RuntimeError("mini-swe-agent query returned no response")
        _append_assistant_to_canonical(canonical_messages, prompt_view, original_length)
        state.note_followup(_signal_from_response(self, response))
        return response

    @wraps(original_get_observation)
    def get_observation_with_posterior_history(
        self: Any, response: dict[str, Any]
    ) -> dict[str, Any]:
        state = _state(self, config)
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
        if isinstance(message, dict) and isinstance(raw_output, str):
            command, path = _causing_action(self, response)
            state.record_observation(
                message,
                raw_output=raw_output,
                causing_command=command,
                causing_path=path,
            )
        return output

    @wraps(original_run)
    def run_with_posterior_history(self: Any, task: str, **kwargs: Any):
        state = _state(self, config)
        state.reset()
        try:
            return original_run(self, task, **kwargs)
        finally:
            if self.messages:
                # This last message is terminal, so it will not be sent to the
                # model again.  It makes aggregate trajectory auditing possible
                # without putting telemetry in an OpenAI message request.
                self.messages[-1]["posterior_history_report"] = state.summary()

    default_agent_class.query = query_with_posterior_history
    default_agent_class.get_observation = get_observation_with_posterior_history
    default_agent_class.run = run_with_posterior_history


def install_hook(config: PosteriorHistoryConfig | None = None) -> bool:
    """Install delayed compaction only when explicitly enabled by environment."""

    global _INSTALLED
    if _INSTALLED:
        return True
    config = config if config is not None else PosteriorHistoryConfig.from_env()
    if config is None:
        return False
    try:
        from minisweagent.agents.default import DefaultAgent
    except ImportError as exc:
        raise RuntimeError(
            "minisweagent is not importable; use the existing SWE-Pruner mini environment"
        ) from exc
    mode = assert_mini_compatible(DefaultAgent)
    _patch(DefaultAgent, config)
    _INSTALLED = True
    LOGGER.info(
        "installed posterior-history hook mode=%s hot_observations=%s method=%s",
        mode,
        config.hot_observations,
        config.method,
    )
    return True
