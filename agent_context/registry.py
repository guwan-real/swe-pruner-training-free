from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agent_context.codecs import (
    CodecRegistry,
    build_legacy_posterior_codec_registry,
    build_typed_codec_registry,
)
from agent_context.config import AgentContextConfig
from agent_context.planner import (
    FullPromptPlanner,
    GlobalBudgetPlanner,
    MinimumViewPlanner,
    PromptPlanner,
)
from agent_context.signals import (
    AllTermsSignalStrategy,
    NoActionSignalProvider,
    NoSignalStrategy,
    PosteriorActionSignalProvider,
    RareTermSignalStrategy,
    SignalProvider,
    SignalStrategy,
)
from agent_context.visibility import (
    BaselineVisibilityPolicy,
    ImmediateVisibilityPolicy,
    PosteriorVisibilityPolicy,
    VisibilityPolicy,
)

CodecFactory = Callable[[AgentContextConfig], CodecRegistry]
SignalProviderFactory = Callable[[AgentContextConfig], SignalProvider]
SignalStrategyFactory = Callable[[AgentContextConfig], SignalStrategy]
PlannerFactory = Callable[[AgentContextConfig], PromptPlanner]
VisibilityFactory = Callable[[AgentContextConfig], VisibilityPolicy]


@dataclass(frozen=True)
class ContextComponents:
    codecs: CodecRegistry
    signal_provider: SignalProvider
    signal_strategy: SignalStrategy
    planner: PromptPlanner
    visibility: VisibilityPolicy


class ComponentRegistry:
    """Factories are explicit so experiments can swap one axis at a time."""

    def __init__(self) -> None:
        self.codec_profiles: dict[str, CodecFactory] = {}
        self.signal_providers: dict[str, SignalProviderFactory] = {}
        self.signal_strategies: dict[str, SignalStrategyFactory] = {}
        self.planners: dict[str, PlannerFactory] = {}
        self.visibility_policies: dict[str, VisibilityFactory] = {}

    def register_codec_profile(self, name: str, factory: CodecFactory) -> None:
        self.codec_profiles[name] = factory

    def register_signal_provider(self, name: str, factory: SignalProviderFactory) -> None:
        self.signal_providers[name] = factory

    def register_signal_strategy(self, name: str, factory: SignalStrategyFactory) -> None:
        self.signal_strategies[name] = factory

    def register_planner(self, name: str, factory: PlannerFactory) -> None:
        self.planners[name] = factory

    def register_visibility(self, name: str, factory: VisibilityFactory) -> None:
        self.visibility_policies[name] = factory

    @staticmethod
    def _build(values: dict[str, Callable[..., Any]], name: str, config: AgentContextConfig):
        try:
            factory = values[name]
        except KeyError as exc:
            choices = ", ".join(sorted(values))
            raise ValueError(f"unknown component {name!r}; choose one of: {choices}") from exc
        return factory(config)

    def components(self, config: AgentContextConfig) -> ContextComponents:
        planner_name = {
            "passthrough": "full",
            "minimum": "minimum",
        }.get(config.planner.mode, "global_budget")
        return ContextComponents(
            codecs=self._build(self.codec_profiles, config.codec_profile, config),
            signal_provider=self._build(self.signal_providers, config.signal_provider, config),
            signal_strategy=self._build(self.signal_strategies, config.signal_strategy, config),
            planner=self._build(self.planners, planner_name, config),
            visibility=self._build(self.visibility_policies, config.timing, config),
        )

    def manifest(self) -> dict[str, tuple[str, ...]]:
        return {
            "codec_profiles": tuple(sorted(self.codec_profiles)),
            "signal_providers": tuple(sorted(self.signal_providers)),
            "signal_strategies": tuple(sorted(self.signal_strategies)),
            "planners": tuple(sorted(self.planners)),
            "timing_policies": tuple(sorted(self.visibility_policies)),
        }


def build_default_registry() -> ComponentRegistry:
    registry = ComponentRegistry()
    registry.register_codec_profile(
        "typed_v1",
        lambda config: build_typed_codec_registry(
            block_max_lines=int(config.codec_options.get("block_max_lines", 16))
        ),
    )
    registry.register_codec_profile(
        "legacy_posterior_v1",
        lambda config: build_legacy_posterior_codec_registry(dict(config.codec_options)),
    )
    registry.register_signal_provider(
        "posterior_action", lambda config: PosteriorActionSignalProvider()
    )
    registry.register_signal_provider("none", lambda config: NoActionSignalProvider())
    registry.register_signal_strategy(
        "rare_terms",
        lambda config: RareTermSignalStrategy(
            max_document_frequency_ratio=float(
                config.codec_options.get("max_document_frequency_ratio", 0.1)
            )
        ),
    )
    registry.register_signal_strategy("all_terms", lambda config: AllTermsSignalStrategy())
    registry.register_signal_strategy("none", lambda config: NoSignalStrategy())
    registry.register_planner("global_budget", lambda config: GlobalBudgetPlanner(config.planner))
    registry.register_planner("full", lambda config: FullPromptPlanner())
    registry.register_planner("minimum", lambda config: MinimumViewPlanner())
    registry.register_visibility("baseline", lambda config: BaselineVisibilityPolicy())
    registry.register_visibility("immediate", lambda config: ImmediateVisibilityPolicy())
    registry.register_visibility("posterior", lambda config: PosteriorVisibilityPolicy())
    return registry


DEFAULT_COMPONENT_REGISTRY = build_default_registry()
