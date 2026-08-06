from __future__ import annotations

import json
from pathlib import Path

from agent_context.adapters.preflight import _load_config
from agent_context.config import AgentContextConfig
from agent_context.registry import DEFAULT_COMPONENT_REGISTRY

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "agent_context_server_arms"


def _raw(arm: str) -> dict:
    return json.loads((CONFIG_ROOT / f"{arm}.json").read_text(encoding="utf-8"))


def _changed_paths(left: object, right: object, prefix: str = "") -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        paths: set[str] = set()
        for key in left.keys() | right.keys():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.add(path)
            else:
                paths.update(_changed_paths(left[key], right[key], path))
        return paths
    return set() if left == right else {prefix}


def test_all_server_arm_configs_are_valid_and_retrieval_is_disabled() -> None:
    for arm in ("R", "D", "E", "F", "F0", "C"):
        config = AgentContextConfig.from_mapping(_raw(arm))
        DEFAULT_COMPONENT_REGISTRY.components(config)
        assert config.track_later_references is False
        assert config.views.include_reference_view is False


def test_legacy_arms_are_true_single_axis_ablations() -> None:
    reference = _raw("R")

    assert _changed_paths(reference, _raw("D")) == {"hot_observations"}
    assert _changed_paths(reference, _raw("E")) == {"signal_provider"}


def test_typed_signal_ablation_and_planner_parity() -> None:
    reference = _raw("R")
    typed = _raw("F")
    typed_without_matcher = _raw("F0")

    assert _changed_paths(typed, typed_without_matcher) == {"signal_strategy"}
    assert typed["planner"] == reference["planner"]
    assert typed["codec_profile"] == "typed_v1"
    assert typed["views"]["include_skeleton_view"] is True


def test_context_limit_arm_is_explicitly_dynamic() -> None:
    reference = _raw("R")
    context_limit = _raw("C")

    assert context_limit["planner"]["mode"] == "context_limit"
    assert context_limit["planner"]["cache_policy"] == "dynamic"
    assert context_limit["planner"]["max_prompt_tokens"] == 32768
    assert context_limit["planner"]["reserve_completion_tokens"] == 4096
    for key, value in reference.items():
        if key != "planner":
            assert context_limit[key] == value


def test_unknown_planner_kind_weight_is_rejected() -> None:
    payload = _raw("F")
    payload["planner"]["kind_weights"] = {"soruce": 999}

    try:
        AgentContextConfig.from_mapping(payload)
    except ValueError as exc:
        assert "unknown planner kind weights" in str(exc)
        assert "soruce" in str(exc)
    else:
        raise AssertionError("unknown planner kind weight was accepted")


def test_reference_arm_neutralizes_old_absolute_admission_gates() -> None:
    options = _raw("R")["codec_options"]

    assert options["min_input_tokens"] == 0
    assert options["min_savings_tokens"] == 1
    assert options["max_retention_ratio"] == 0.999


def test_reference_view_is_rejected_in_per_kind_overrides(tmp_path: Path) -> None:
    payload = _raw("F")
    payload["view_overrides"] = {"source": {"include_reference_view": True}}
    path = tmp_path / "unsafe.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        _load_config(path)
    except ValueError as exc:
        assert "reference views" in str(exc)
    else:
        raise AssertionError("per-kind reference view override was accepted")


def test_invalid_view_override_schema_is_rejected() -> None:
    payload = _raw("F")
    payload["view_overrides"] = {"unknown_kind": {"include_focused_view": True}}
    try:
        AgentContextConfig.from_mapping(payload)
    except ValueError as exc:
        assert "unknown view override kinds" in str(exc)
    else:
        raise AssertionError("unknown view override kind was accepted")

    payload = _raw("F")
    payload["view_overrides"] = {"source": {"unknown_field": True}}
    try:
        AgentContextConfig.from_mapping(payload)
    except TypeError as exc:
        assert "unknown_field" in str(exc)
    else:
        raise AssertionError("unknown view override field was accepted")


def test_unknown_component_options_are_rejected_by_the_registry() -> None:
    payload = _raw("F")
    payload["codec_options"]["block_max_line"] = 8
    try:
        DEFAULT_COMPONENT_REGISTRY.components(AgentContextConfig.from_mapping(payload))
    except ValueError as exc:
        assert "unknown codec profile options" in str(exc)
        assert "block_max_line" in str(exc)
    else:
        raise AssertionError("unknown codec option was accepted")

    payload = _raw("F")
    payload["signal_options"] = {"max_document_frequency": 0.2}
    try:
        DEFAULT_COMPONENT_REGISTRY.components(AgentContextConfig.from_mapping(payload))
    except ValueError as exc:
        assert "unknown signal strategy options" in str(exc)
        assert "max_document_frequency" in str(exc)
    else:
        raise AssertionError("unknown signal option was accepted")
