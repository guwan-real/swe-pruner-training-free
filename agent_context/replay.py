from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from agent_context.config import AgentContextConfig
from agent_context.engine import ContextEngine
from agent_context.models import ActionEvent, PromptManifest


@dataclass(frozen=True)
class ReplayResult:
    report: Mapping[str, Any]
    manifests: tuple[PromptManifest, ...]
    canonical_messages: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "report": dict(self.report),
            "manifests": [manifest.to_dict() for manifest in self.manifests],
            "canonical_messages": list(self.canonical_messages),
        }


def replay_trace(
    payload: Mapping[str, Any],
    config: AgentContextConfig | Mapping[str, Any] | None = None,
) -> ReplayResult:
    """Replay structured agent events without invoking an agent model."""

    task_id = str(payload.get("task_id", "replay-task"))
    task_text = str(payload.get("task_text", ""))
    events = payload.get("events", ())
    if not isinstance(events, list):
        raise ValueError("trace events must be a list")
    engine = ContextEngine(config)
    engine.start_task(task_id, task_text=task_text)
    messages: list[dict[str, Any]] = []
    for index, event_value in enumerate(events, start=1):
        if not isinstance(event_value, Mapping):
            raise ValueError(f"event {index} must be an object")
        event = dict(event_value)
        event_type = str(event.get("type", ""))
        if event_type == "message":
            message = event.get("message")
            if not isinstance(message, Mapping):
                raise ValueError(f"event {index}: message must be an object")
            messages.append(dict(message))
            continue
        if event_type == "observation":
            visible = event.get("visible_content")
            if not isinstance(visible, str):
                raise ValueError(f"event {index}: visible_content must be a string")
            message_value = event.get("message")
            if message_value is None:
                message = {"role": "user", "content": visible}
            elif isinstance(message_value, Mapping):
                message = dict(message_value)
            else:
                raise ValueError(f"event {index}: message must be an object")
            messages.append(message)
            engine.ingest_observation(
                message,
                visible_content=visible,
                raw_content=(
                    None if event.get("raw_content") is None else str(event["raw_content"])
                ),
                causing_action=str(event.get("causing_action", "")),
                path=str(event.get("path", "")),
                kind=(None if event.get("kind") is None else str(event["kind"])),
                observation_id=(
                    None if event.get("observation_id") is None else str(event["observation_id"])
                ),
                metadata=(
                    dict(event["metadata"]) if isinstance(event.get("metadata"), Mapping) else None
                ),
            )
            continue
        if event_type == "prompt":
            engine.build_prompt(messages)
            continue
        if event_type == "action":
            message_value = event.get("message")
            if isinstance(message_value, Mapping):
                messages.append(dict(message_value))
            engine.observe_action(
                ActionEvent(
                    step=int(event.get("step", engine.step + 1)),
                    command=str(event.get("command", "")),
                    context_focus_question=str(event.get("context_focus_question", "")),
                    response_content=str(event.get("response_content", "")),
                    metadata=(
                        dict(event["metadata"])
                        if isinstance(event.get("metadata"), Mapping)
                        else {}
                    ),
                )
            )
            continue
        if event_type == "memory":
            operation = str(event.get("operation", ""))
            observation_id = str(event.get("observation_id", ""))
            if operation == "pin":
                engine.pin(observation_id)
            elif operation == "unpin":
                engine.unpin(observation_id)
            elif operation == "read":
                engine.read(
                    observation_id,
                    start_line=int(event.get("start_line", 1)),
                    end_line=(None if event.get("end_line") is None else int(event["end_line"])),
                )
            else:
                raise ValueError(f"event {index}: unknown memory operation {operation!r}")
            continue
        raise ValueError(f"event {index}: unknown type {event_type!r}")
    report = engine.report()
    return ReplayResult(
        report=report,
        manifests=engine.manifests,
        canonical_messages=tuple(messages),
    )
