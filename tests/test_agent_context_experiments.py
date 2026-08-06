from __future__ import annotations

from agent_context.config import AgentContextConfig
from agent_context.experiments import expand_ablation_matrix
from agent_context.registry import DEFAULT_COMPONENT_REGISTRY


def test_default_registry_exposes_orthogonal_experiment_components() -> None:
    manifest = DEFAULT_COMPONENT_REGISTRY.manifest()

    assert {"typed_v1", "legacy_posterior_v1"}.issubset(manifest["codec_profiles"])
    assert {"none", "posterior_action"}.issubset(manifest["signal_providers"])
    assert {"none", "all_terms", "rare_terms"}.issubset(manifest["signal_strategies"])
    assert {"baseline", "immediate", "posterior"}.issubset(manifest["timing_policies"])


def test_ablation_matrix_changes_one_nested_axis_without_mutating_base() -> None:
    base = AgentContextConfig()
    arms = expand_ablation_matrix(
        base,
        {
            "hot_observations": [1, 2],
            "planner.target_retention": [0.5, 0.7],
            "signal_strategy": ["rare_terms", "none"],
        },
        prefix="posterior-typed",
    )

    assert len(arms) == 8
    assert {arm.config.hot_observations for arm in arms} == {1, 2}
    assert {arm.config.planner.target_retention for arm in arms} == {0.5, 0.7}
    assert base.hot_observations == 2
    assert base.planner.target_retention == 0.6
