from __future__ import annotations

from agent_context import ActionEvent, AgentContextConfig, ContextEngine
from agent_context.memory import MemoryToolRequest, ObservationMemoryTools


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


def _config(**values) -> AgentContextConfig:
    payload = {
        "hot_observations": 1,
        "track_later_references": False,
        "planner": {"mode": "retention", "target_retention": 0.55},
    }
    payload.update(values)
    return AgentContextConfig.from_mapping(payload)


def test_posterior_lifecycle_reads_full_before_compacting_cold_history() -> None:
    engine = ContextEngine(_config())
    engine.start_task("task-1", task_text="fix resolve_model")
    source = _source()
    first = {"role": "user", "content": f"Observation: {source}", "extra": {"x": []}}
    messages = [{"role": "system", "content": "system"}, first]
    first_content = first["content"]
    first_observation = engine.ingest_observation(
        first,
        visible_content=source,
        causing_action="sed -n '1,999p' model.py",
        path="model.py",
    )

    first_build = engine.build_prompt(messages)
    assert first_build.manifest.entries[0].selected_level == "full"
    assert first_build.messages[-1]["content"] == first_content
    engine.observe_action(ActionEvent(step=engine.step + 1, command="rg -n resolve_model model.py"))

    second = {"role": "user", "content": f"Observation: {source}"}
    messages.append(second)
    engine.ingest_observation(
        second,
        visible_content=source,
        causing_action="sed -n '1,999p' model.py",
        path="model.py",
    )
    second_build = engine.build_prompt(messages)
    entries = {entry.observation_id: entry for entry in second_build.manifest.entries}

    assert entries[first_observation.id].tier == "cold"
    assert entries[first_observation.id].selected_level != "full"
    assert entries[engine.store.values()[-1].observation.id].selected_level == "full"
    assert first["content"] == first_content
    assert second_build.manifest.estimated_tokens_saved > 0


def test_prompt_build_is_deeply_detached_and_strips_framework_telemetry() -> None:
    engine = ContextEngine(_config(timing="baseline"))
    engine.start_task("task")
    source = _source()
    message = {
        "role": "user",
        "content": source,
        "extra": {"nested": ["canonical"]},
        "agent_context_stats": {"status": "tracked"},
    }
    engine.ingest_observation(message, visible_content=source, path="model.py")

    build = engine.build_prompt([message])
    build.messages[0]["extra"]["nested"].append("client")

    assert message["extra"]["nested"] == ["canonical"]
    assert "agent_context_stats" not in build.messages[0]


def test_memory_tools_read_search_pin_and_append_without_a_model() -> None:
    engine = ContextEngine(_config())
    engine.start_task("task")
    source = _source()
    message = {"role": "user", "content": source}
    observation = engine.ingest_observation(message, visible_content=source, path="model.py")
    tools = ObservationMemoryTools(engine)

    search = tools.execute(
        MemoryToolRequest(
            operation="search",
            observation_id=observation.id,
            arguments={"pattern": "resolve_model"},
        )
    )
    read = tools.execute(
        MemoryToolRequest(
            operation="read",
            observation_id=observation.id,
            arguments={"start_line": 1, "end_line": 2},
        )
    )
    tools.execute(MemoryToolRequest(operation="pin", observation_id=observation.id, arguments={}))

    assert "resolve_model" in search.content
    assert len(read.content.splitlines()) == 2
    assert engine.store.get(observation.id).pinned is True
    assert engine.report()["model_forward_count"] == 0
    assert engine.report()["llm_token_count"] == 0


def test_freeze_on_cold_preserves_committed_prefix_across_later_plans() -> None:
    engine = ContextEngine(_config(hot_observations=1))
    engine.start_task("cache-stability")
    source = _source()
    messages: list[dict[str, str]] = []

    first = {"role": "user", "content": source}
    messages.append(first)
    first_observation = engine.ingest_observation(first, visible_content=source, path="model.py")
    engine.build_prompt(messages)
    engine.observe_action(ActionEvent(step=engine.step + 1, command="rg resolve_model model.py"))

    second = {"role": "user", "content": source}
    messages.append(second)
    engine.ingest_observation(second, visible_content=source, path="model.py")
    committed_build = engine.build_prompt(messages)
    first_entry = committed_build.manifest.entries[0]
    committed_content = committed_build.messages[0]["content"]

    assert first_entry.committed is True
    assert first_entry.selection_changed is True
    assert first_entry.selected_level != "full"
    assert committed_build.manifest.earliest_context_change_message_index == 0

    # This later signal would change the old focused view under dynamic replanning.
    engine.observe_action(ActionEvent(step=engine.step + 1, command="rg helper_29 model.py"))
    third = {"role": "user", "content": source}
    messages.append(third)
    engine.ingest_observation(third, visible_content=source, path="model.py")
    later_build = engine.build_prompt(messages)
    later_first_entry = later_build.manifest.entries[0]

    assert later_build.messages[0]["content"] == committed_content
    assert later_first_entry.view_fingerprint == first_entry.view_fingerprint
    assert later_first_entry.selection_changed is False
    assert later_build.manifest.earliest_context_change_message_index == 1

    tools = ObservationMemoryTools(engine)
    read = tools.execute(
        MemoryToolRequest(
            operation="read",
            observation_id=first_observation.id,
            arguments={"start_line": 1, "end_line": 4},
        )
    )
    stable_build = engine.build_prompt(messages)

    assert read.metadata["delivery"] == "append-only"
    assert read.metadata["historical_slot_rewritten"] is False
    assert stable_build.messages[0]["content"] == committed_content
    assert stable_build.manifest.entries[0].selection_changed is False
    assert stable_build.manifest.earliest_context_change_message_index is None
    assert engine.store.get(first_observation.id).view_switch_count == 1


def test_dynamic_cache_policy_remains_available_for_ablation() -> None:
    engine = ContextEngine(
        _config(
            hot_observations=1,
            planner={
                "mode": "retention",
                "cache_policy": "dynamic",
                "target_retention": 0.55,
            },
        )
    )
    engine.start_task("dynamic-control")
    source = _source()
    messages: list[dict[str, str]] = []

    first = {"role": "user", "content": source}
    messages.append(first)
    first_observation = engine.ingest_observation(first, visible_content=source, path="model.py")
    engine.build_prompt(messages)
    engine.observe_action(ActionEvent(step=engine.step + 1, command="rg resolve_model model.py"))
    second = {"role": "user", "content": source}
    messages.append(second)
    engine.ingest_observation(second, visible_content=source, path="model.py")
    engine.build_prompt(messages)

    assert engine.store.get(first_observation.id).committed_view is None
    assert engine.components.planner.name == "global_budget_greedy_v1"


def test_baseline_timing_materializes_exact_canonical_observations() -> None:
    engine = ContextEngine(_config(timing="baseline"))
    engine.start_task("task")
    source = _source()
    messages = []
    for _ in range(3):
        message = {"role": "user", "content": f"<output>{source}</output>"}
        messages.append(message)
        engine.ingest_observation(message, visible_content=source)
        engine.build_prompt(messages)
        engine.observe_action(ActionEvent(step=engine.step + 1, command="true"))

    build = engine.build_prompt(messages)

    assert [message["content"] for message in build.messages] == [
        message["content"] for message in messages
    ]
    assert all(entry.selected_level == "full" for entry in build.manifest.entries)
