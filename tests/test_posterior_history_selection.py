from __future__ import annotations

from math import ceil

from posterior_history_pruning.protocol import PosteriorHistoryConfig, PosteriorSignal
from posterior_history_pruning.selection import (
    TOKEN_ESTIMATOR,
    compact_after_followup,
    estimate_tokens,
)
from posterior_history_pruning.state import (
    PosteriorHistoryState,
    locate_observation_boundary,
)


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


def test_token_proxy_corrects_lexical_undercount_for_long_code_tokens() -> None:
    text = "\n".join(f"very_long_generated_identifier_{index:04d}" for index in range(300))

    assert estimate_tokens(text) >= ceil(len(text) / 4)
    assert estimate_tokens(text) > 1500


def test_common_response_words_do_not_expand_posterior_match_to_every_block() -> None:
    source = _source()
    result = compact_after_followup(
        source,
        causing_command="sed -n '1,999p' model.py",
        causing_path="model.py",
        posterior=PosteriorSignal(
            command="rg -n helper_5 model.py",
            context_focus_question="Inspect helper_5.",
            response_content=(
                "I will inspect the config value and result in the helper function before editing."
            ),
        ),
        config=_config(),
    )

    assert result.status == "compacted"
    assert result.matched_block_count > 0
    assert result.matched_block_count < result.block_count
    assert result.selected_block_count < result.block_count


def test_no_safe_reduction_reports_whether_hard_skeleton_consumed_every_block() -> None:
    source = "\n".join(f"def helper_{index}(): pass" for index in range(300))
    result = compact_after_followup(
        source,
        causing_command="cat dense.py",
        causing_path="dense.py",
        posterior=PosteriorSignal(command="true"),
        config=_config(method="safe"),
    )

    assert result.status == "skipped"
    assert result.reason == "no-safe-reduction-hard-skeleton"
    assert result.hard_block_count == result.block_count


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
    assert first["posterior_history_stats"]["token_estimator"] == TOKEN_ESTIMATOR


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


def _official_truncated_message(raw_output: str) -> str:
    return (
        "<returncode>0</returncode>\n"
        "<warning>The output of your last command was too long.</warning>\n"
        "<output_head>\n"
        f"{raw_output[:5000]}\n"
        "</output_head>\n"
        "<elided_chars>\n"
        f"{len(raw_output) - 10000} characters elided\n"
        "</elided_chars>\n"
        "<output_tail>\n"
        f"{raw_output[-5000:]}\n"
        "</output_tail>"
    )


def test_official_head_tail_template_is_tracked_and_compacted_as_visible_text() -> None:
    state = PosteriorHistoryState(_config())
    source = _source()
    rendered_content = _official_truncated_message(source)
    first = {"role": "user", "content": rendered_content}
    second = {"role": "user", "content": f"Observation: {source}"}
    messages = [{"role": "system", "content": "system"}, first]

    assert state.record_observation(
        first,
        raw_output=source,
        causing_command="sed -n '1,999p' model.py",
        causing_path="model.py",
    )
    assert first["posterior_history_stats"]["boundary_mode"] == "official-head-tail"
    assert first["posterior_history_stats"]["source_output_chars"] == len(source)
    assert first["posterior_history_stats"]["visible_output_chars"] < len(source)
    state.note_followup(PosteriorSignal(command="rg -n helper_5 model.py"))

    messages.append(second)
    state.record_observation(
        second,
        raw_output=source,
        causing_command="sed -n '1,999p' model.py",
        causing_path="model.py",
    )
    cold_view = state.render(messages)

    assert "posterior_history_compaction" in cold_view[1]["content"]
    assert '<output_posterior_view source="official-head-tail">' in cold_view[1]["content"]
    assert "<output_head>" not in cold_view[1]["content"]
    assert "def helper_5" in cold_view[1]["content"]
    assert first["content"] == rendered_content


def test_malformed_head_tail_template_fails_open_with_visible_telemetry() -> None:
    state = PosteriorHistoryState(_config())
    source = _source()
    malformed = {"role": "user", "content": f"<output_head>{source[:5000]}</output_head>"}

    assert not state.record_observation(
        malformed,
        raw_output=source,
        causing_command="cat model.py",
        causing_path="model.py",
    )
    stats = malformed["posterior_history_stats"]
    assert stats["status"] == "untracked"
    assert stats["reason"] == "rendered-output-boundary-not-found"
    assert state.summary()["untracked_observations"] == 1
    assert state.render([malformed])[0]["content"] == malformed["content"]


def test_head_tail_boundary_rejects_mismatched_raw_anchors() -> None:
    source = _source()
    content = _official_truncated_message(source)

    assert locate_observation_boundary(content, "x" * len(source)) is None


def test_official_output_wrapper_does_not_confuse_output_with_returncode() -> None:
    content = "<returncode>0</returncode>\n<output>\n0\n</output>"

    boundary = locate_observation_boundary(content, "0")

    assert boundary is not None
    assert boundary.mode == "official-output"
    assert boundary.prefix.endswith("<output>\n")
    assert boundary.prefix + boundary.selection_output + boundary.suffix == content


def test_empty_output_is_not_reported_as_an_unrecognized_template() -> None:
    state = PosteriorHistoryState(_config())
    message = {"role": "user", "content": "<returncode>0</returncode>\n<output>\n</output>"}

    assert not state.record_observation(
        message,
        raw_output="",
        causing_command="true",
        causing_path="",
    )
    assert "posterior_history_stats" not in message
    assert state.summary()["observations_seen"] == 0
    assert state.summary()["untracked_observations"] == 0
