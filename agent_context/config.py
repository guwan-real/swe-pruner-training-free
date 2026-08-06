from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from agent_context.codecs.base import ViewGenerationConfig
from agent_context.models import ObservationKind


@dataclass(frozen=True)
class PlannerConfig:
    mode: str = "retention"
    cache_policy: str = "freeze_on_cold"
    target_retention: float = 0.6
    observation_budget: int | None = None
    max_prompt_tokens: int | None = None
    reserve_completion_tokens: int = 0
    recency_weight: float = 1.0
    relevance_weight: float = 2.0
    kind_weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in {
            "passthrough",
            "minimum",
            "retention",
            "fixed",
            "context_limit",
        }:
            raise ValueError(
                "planner mode must be passthrough, minimum, retention, fixed, or context_limit"
            )
        if self.cache_policy not in {"freeze_on_cold", "dynamic"}:
            raise ValueError("planner cache_policy must be freeze_on_cold or dynamic")
        if not 0.0 < self.target_retention <= 1.0:
            raise ValueError("target_retention must be in (0, 1]")
        if self.observation_budget is not None and self.observation_budget < 0:
            raise ValueError("observation_budget must be non-negative")
        if self.max_prompt_tokens is not None and self.max_prompt_tokens < 1:
            raise ValueError("max_prompt_tokens must be positive")
        if self.reserve_completion_tokens < 0:
            raise ValueError("reserve_completion_tokens must be non-negative")
        if (
            self.max_prompt_tokens is not None
            and self.reserve_completion_tokens >= self.max_prompt_tokens
        ):
            raise ValueError("reserve_completion_tokens must be less than max_prompt_tokens")
        if self.recency_weight < 0 or self.relevance_weight < 0:
            raise ValueError("planner weights must be non-negative")
        normalized_kind_weights = {
            str(kind): float(weight) for kind, weight in self.kind_weights.items()
        }
        valid_kinds = {kind.value for kind in ObservationKind}
        unknown_kinds = sorted(set(normalized_kind_weights).difference(valid_kinds))
        if unknown_kinds:
            raise ValueError("unknown planner kind weights: " + ", ".join(unknown_kinds))
        if any(weight <= 0 for weight in normalized_kind_weights.values()):
            raise ValueError("planner kind weights must be positive")
        object.__setattr__(self, "kind_weights", normalized_kind_weights)
        if self.mode == "fixed" and self.observation_budget is None:
            raise ValueError("fixed planner mode requires observation_budget")
        if self.mode == "context_limit" and self.max_prompt_tokens is None:
            raise ValueError("context_limit planner mode requires max_prompt_tokens")


@dataclass(frozen=True)
class AgentContextConfig:
    timing: str = "posterior"
    hot_observations: int = 2
    codec_profile: str = "typed_v1"
    signal_provider: str = "posterior_action"
    signal_strategy: str = "rare_terms"
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    views: ViewGenerationConfig = field(default_factory=ViewGenerationConfig)
    view_overrides: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    codec_options: Mapping[str, Any] = field(default_factory=dict)
    signal_options: Mapping[str, Any] = field(default_factory=dict)
    include_task_signal: bool = False
    include_causing_action_signal: bool = False
    track_later_references: bool = True
    telemetry_fields: tuple[str, ...] = (
        "agent_context_manifest",
        "agent_context_report",
        "agent_context_stats",
    )

    def __post_init__(self) -> None:
        if self.timing not in {"baseline", "immediate", "posterior"}:
            raise ValueError("timing must be baseline, immediate, or posterior")
        if self.hot_observations < 0:
            raise ValueError("hot_observations must be non-negative")
        if not self.codec_profile:
            raise ValueError("codec_profile must not be empty")
        if not self.signal_provider or not self.signal_strategy:
            raise ValueError("signal components must not be empty")
        normalized_overrides = {str(key): dict(value) for key, value in self.view_overrides.items()}
        valid_kinds = {kind.value for kind in ObservationKind}
        unknown_kinds = sorted(set(normalized_overrides).difference(valid_kinds))
        if unknown_kinds:
            raise ValueError("unknown view override kinds: " + ", ".join(unknown_kinds))
        base_views = asdict(self.views)
        for kind, override in normalized_overrides.items():
            ViewGenerationConfig(**(base_views | override))
        object.__setattr__(self, "view_overrides", normalized_overrides)
        object.__setattr__(self, "codec_options", dict(self.codec_options))
        object.__setattr__(self, "signal_options", dict(self.signal_options))

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None = None) -> AgentContextConfig:
        payload = dict(values or {})
        planner_value = payload.pop("planner", {})
        views_value = payload.pop("views", {})
        planner = (
            planner_value
            if isinstance(planner_value, PlannerConfig)
            else PlannerConfig(**dict(planner_value))
        )
        views = (
            views_value
            if isinstance(views_value, ViewGenerationConfig)
            else ViewGenerationConfig(**dict(views_value))
        )
        if "telemetry_fields" in payload:
            payload["telemetry_fields"] = tuple(payload["telemetry_fields"])
        return cls(planner=planner, views=views, **payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def views_for_kind(self, kind: str) -> ViewGenerationConfig:
        override = self.view_overrides.get(kind)
        if not override:
            return self.views
        values = asdict(self.views)
        values.update(dict(override))
        return ViewGenerationConfig(**values)
