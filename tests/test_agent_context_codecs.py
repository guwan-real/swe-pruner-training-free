from __future__ import annotations

from agent_context.codecs import build_typed_codec_registry, classify_observation
from agent_context.codecs.base import ViewGenerationConfig
from agent_context.estimation import DEFAULT_ESTIMATOR
from agent_context.models import ContextSignal, Observation, ObservationKind, ViewLevel
from agent_context.signals import RareTermSignalStrategy


def _source() -> str:
    values: list[str] = []
    for index in range(20):
        name = "resolve_model" if index == 10 else f"helper_{index}"
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


def _observation(kind: ObservationKind, content: str, *, path: str = "") -> Observation:
    return Observation(
        id="obs-1",
        task_id="task",
        step=1,
        raw_content=content,
        visible_content=content,
        path=path,
        kind=kind,
    )


def test_classification_routes_tool_outputs_to_typed_codecs() -> None:
    assert (
        classify_observation("diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py")
        == ObservationKind.DIFF
    )
    assert (
        classify_observation(
            'Traceback (most recent call last):\n  File "a.py", line 1\nValueError: bad'
        )
        == ObservationKind.TRACEBACK
    )
    assert (
        classify_observation("a.py:10:resolve_model", command="rg resolve_model")
        == ObservationKind.SEARCH
    )
    assert classify_observation(_source(), path="model.py") == ObservationKind.SOURCE


def test_source_codec_produces_multiresolution_signal_guided_views() -> None:
    observation = _observation(ObservationKind.SOURCE, _source(), path="model.py")
    codec = build_typed_codec_registry(block_max_lines=8).get(ObservationKind.SOURCE)
    document = codec.parse(observation)
    signal = ContextSignal(
        provider="next_action.command",
        text="rg -n resolve_model model.py",
        step=2,
        weight=2.0,
    )
    match = RareTermSignalStrategy().score(document, [signal])

    views = codec.generate_views(
        observation,
        document,
        [signal],
        match,
        DEFAULT_ESTIMATOR,
        ViewGenerationConfig(),
    )
    by_level = {view.level: view for view in views}

    assert {ViewLevel.SKELETON, ViewLevel.FOCUSED, ViewLevel.FULL}.issubset(by_level)
    assert "resolve_model" in by_level[ViewLevel.FOCUSED].content
    assert by_level[ViewLevel.SKELETON].token_count < by_level[ViewLevel.FULL].token_count
    assert by_level[ViewLevel.FOCUSED].metadata["matched_terms"]


def test_reference_view_is_explicit_opt_in() -> None:
    observation = _observation(
        ObservationKind.SEARCH, "\n".join(f"a.py:{i}: value" for i in range(30))
    )
    codec = build_typed_codec_registry().get(ObservationKind.SEARCH)
    document = codec.parse(observation)
    match = RareTermSignalStrategy().score(document, [])

    default_views = codec.generate_views(
        observation, document, [], match, DEFAULT_ESTIMATOR, ViewGenerationConfig()
    )
    reference_views = codec.generate_views(
        observation,
        document,
        [],
        match,
        DEFAULT_ESTIMATOR,
        ViewGenerationConfig(include_reference_view=True),
    )

    assert ViewLevel.REFERENCE not in {view.level for view in default_views}
    assert ViewLevel.REFERENCE in {view.level for view in reference_views}


def test_focused_view_fits_soft_expansion_to_output_cap() -> None:
    observation = _observation(ObservationKind.SOURCE, _source(), path="model.py")
    codec = build_typed_codec_registry(block_max_lines=8).get(ObservationKind.SOURCE)
    document = codec.parse(observation)
    signal = ContextSignal(provider="next_action.command", text="resolve_model", step=2, weight=2.0)
    match = RareTermSignalStrategy().score(document, [signal])

    views = codec.generate_views(
        observation,
        document,
        [signal],
        match,
        DEFAULT_ESTIMATOR,
        ViewGenerationConfig(max_output_chars=1800),
    )
    focused = next(view for view in views if view.level == ViewLevel.FOCUSED)

    assert len(focused.content) <= 1800
    assert "resolve_model" in focused.content


def test_diff_codec_only_allows_the_full_view() -> None:
    diff = "\n".join(
        ["diff --git a/a.py b/a.py", "--- a/a.py", "+++ b/a.py"]
        + [f"+changed {index}" for index in range(100)]
    )
    observation = _observation(ObservationKind.DIFF, diff)
    codec = build_typed_codec_registry().get(ObservationKind.DIFF)
    document = codec.parse(observation)
    match = RareTermSignalStrategy().score(document, [])

    views = codec.generate_views(
        observation, document, [], match, DEFAULT_ESTIMATOR, ViewGenerationConfig()
    )

    assert [view.level for view in views] == [ViewLevel.FULL]


def test_test_log_codec_preserves_failures_as_mandatory_evidence() -> None:
    log = "\n".join(
        [f"test_{index} PASSED" for index in range(50)]
        + ["==== FAILURES ====", "FAILED test_model.py::test_resolve", "AssertionError: bad"]
    )
    observation = _observation(ObservationKind.TEST_LOG, log)
    codec = build_typed_codec_registry().get(ObservationKind.TEST_LOG)
    document = codec.parse(observation)

    assert any(unit.mandatory and "FAILED" in unit.text for unit in document.units)
    assert any(unit.mandatory and "AssertionError" in unit.text for unit in document.units)
