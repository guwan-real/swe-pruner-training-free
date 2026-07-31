from __future__ import annotations

from posterior_history_pruning.protocol import PosteriorHistoryConfig, PosteriorSignal
from posterior_history_pruning.selection import compact_after_followup
from posterior_history_pruning.state import PosteriorHistoryState


def _source() -> str:
    chunks: list[str] = []
    for index in range(90):
        name = "resolve_model" if index == 44 else f"helper_{index}"
        chunks.append(
            f"def {name}(config):\n"
            f"    value_{index} = config.get('key_{index}')\n"
            f"    if value_{index} is None:\n"
            f"        raise ValueError('key_{index}')\n"
            f"    return value_{index}\n"
        )
    return "\n".join(chunks)


def _config(**overrides: object) -> PosteriorHistoryConfig:
    values: dict[str, object] = {
        "hot_observations": 1,
        "min_input_tokens": 0,
        "min_savings_tokens": 1,
        "max_retention_ratio": 0.99,
        "block_max_lines": 8,
        "max_output_chars": 50000,
        "method": "adaptive",
    }
    values.update(overrides)
    return PosteriorHistoryConfig(**values)  # type: ignore[arg-type]


def test_posterior_selector_keeps_followup_symbol_and_compacts_old_source() -> None:
    source = _source()
    result = compact_after_followup(
        source,
        causing_command="sed -n '1,999p' model.py",
        causing_path="model.py",
        posterior=PosteriorSignal(
            command="rg -n 'resolve_model' model.py",
            context_focus_question="Where is resolve_model validated?",
        ),
        config=_config(),
    )

    assert result.status == "compacted"
    assert "def resolve_model" in result.text
    assert result.left_token_cnt < result.origin_token_cnt
    assert result.retention_ratio < 0.99
    assert "posterior_history_compaction" in result.text


def test_selector_fails_open_without_an_actual_posterior_match() -> None:
    source = _source()
    result = compact_after_followup(
        source,
        causing_command="sed -n '1,999p' model.py",
        causing_path="model.py",
        posterior=PosteriorSignal(command="rg -n 'unrelated_symbol' other.py"),
        config=_config(),
    )

    assert result.status == "skipped"
    assert result.reason == "no-posterior-match"
    assert result.text == source


def test_diff_is_never_compacted() -> None:
    diff = "\n".join(
        ["diff --git a/a.py b/a.py", "--- a/a.py", "+++ b/a.py"]
        + [f"+line {index}" for index in range(400)]
    )
    result = compact_after_followup(
        diff,
        causing_command="git diff",
        causing_path="a.py",
        posterior=PosteriorSignal(command="git diff -- a.py"),
        config=_config(),
    )

    assert result.status == "skipped"
    assert result.reason == "diff-is-never-compacted"
    assert result.text == diff


def test_state_keeps_hot_observation_full_and_never_mutates_canonical_content() -> None:
    state = PosteriorHistoryState(_config())
    source = _source()
    first = {"role": "user", "content": f"Observation: {source}"}
    second = {"role": "user", "content": f"Observation: {source}"}
    messages = [{"role": "system", "content": "system"}, first]
    state.record_observation(
        first,
        raw_output=source,
        causing_command="sed -n '1,999p' model.py",
        causing_path="model.py",
    )
    state.note_followup(PosteriorSignal(command="rg -n resolve_model model.py"))

    # The newest observation remains verbatim even after its posterior action exists.
    hot_view = state.render(messages)
    assert hot_view[-1]["content"] == first["content"]

    messages.append(second)
    state.record_observation(
        second,
        raw_output=source,
        causing_command="sed -n '1,999p' model.py",
        causing_path="model.py",
    )
    state.note_followup(PosteriorSignal(command="rg -n resolve_model model.py"))
    cold_view = state.render(messages)

    assert "posterior_history_compaction" in cold_view[1]["content"]
    assert cold_view[2]["content"] == second["content"]
    assert first["content"] == f"Observation: {source}"
    assert "posterior_history_stats" in first
    assert "posterior_history_stats" not in cold_view[1]
    assert first["posterior_history_stats"]["prompt_compaction_count"] == 1
    assert first["posterior_history_stats"]["total_prompt_tokens_saved"] > 0


def test_rendered_prompt_is_fully_detached_from_canonical_message_metadata() -> None:
    state = PosteriorHistoryState(_config())
    canonical = {
        "role": "user",
        "content": "Observation: short",
        "extra": {"request_metadata": ["original"]},
    }

    rendered = state.render([canonical])
    rendered[0]["extra"]["request_metadata"].append("client-mutation")

    assert canonical["extra"]["request_metadata"] == ["original"]
