from __future__ import annotations

from agent_context import ActionEvent, AgentContextConfig, ContextEngine
from posterior_history_pruning.protocol import PosteriorHistoryConfig, PosteriorSignal
from posterior_history_pruning.selection import compact_after_followup


def _source() -> str:
    values: list[str] = []
    for index in range(40):
        name = "resolve_model" if index == 20 else f"helper_{index}"
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


def test_legacy_profile_reuses_existing_selector_output_exactly() -> None:
    config = AgentContextConfig.from_mapping(
        {
            "timing": "posterior",
            "hot_observations": 1,
            "codec_profile": "legacy_posterior_v1",
            "planner": {"mode": "minimum"},
            "codec_options": {
                "method": "adaptive",
                "min_input_tokens": 0,
                "min_savings_tokens": 1,
                "max_retention_ratio": 0.99,
                "block_max_lines": 8,
                "max_output_chars": 50000,
            },
            "track_later_references": False,
        }
    )
    source = _source()
    signal = PosteriorSignal(command="rg -n resolve_model model.py")
    expected = compact_after_followup(
        source,
        causing_command="sed -n '1,999p' model.py",
        causing_path="model.py",
        posterior=signal,
        config=PosteriorHistoryConfig(
            hot_observations=1,
            min_input_tokens=0,
            min_savings_tokens=1,
            max_retention_ratio=0.99,
            block_max_lines=8,
            max_output_chars=50000,
            method="adaptive",
        ),
    )
    assert expected.status == "compacted"

    engine = ContextEngine(config)
    engine.start_task("legacy")
    first = {"role": "user", "content": source}
    messages = [first]
    observation = engine.ingest_observation(
        first,
        visible_content=source,
        causing_action="sed -n '1,999p' model.py",
        path="model.py",
    )
    engine.build_prompt(messages)
    engine.observe_action(ActionEvent(step=engine.step + 1, command=signal.command))
    second = {"role": "user", "content": source}
    messages.append(second)
    engine.ingest_observation(second, visible_content=source, path="model.py")

    build = engine.build_prompt(messages)
    entry = next(
        value for value in build.manifest.entries if value.observation_id == observation.id
    )

    assert entry.selected_level == "focused"
    assert build.messages[0]["content"] == expected.text
    assert first["content"] == source
