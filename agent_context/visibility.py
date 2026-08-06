from __future__ import annotations

from typing import Protocol

from agent_context.models import (
    LifecycleStage,
    MemoryTier,
    ObservationRuntime,
    ViewLevel,
)


class VisibilityPolicy(Protocol):
    name: str

    def allowed_views(self, runtime: ObservationRuntime) -> tuple[ViewLevel, ...]: ...


def _available(runtime: ObservationRuntime) -> tuple[ViewLevel, ...]:
    return tuple(sorted(runtime.views))


class BaselineVisibilityPolicy:
    name = "baseline"

    def allowed_views(self, runtime: ObservationRuntime) -> tuple[ViewLevel, ...]:
        return (ViewLevel.FULL,)


class ImmediateVisibilityPolicy:
    name = "immediate"

    def allowed_views(self, runtime: ObservationRuntime) -> tuple[ViewLevel, ...]:
        if runtime.pinned or runtime.tier == MemoryTier.PINNED:
            return (ViewLevel.FULL,)
        return _available(runtime)


class PosteriorVisibilityPolicy:
    name = "posterior"

    def allowed_views(self, runtime: ObservationRuntime) -> tuple[ViewLevel, ...]:
        if runtime.stage != LifecycleStage.ENRICHED:
            return (ViewLevel.FULL,)
        if runtime.pinned or runtime.tier in {
            MemoryTier.HOT,
            MemoryTier.PINNED,
        }:
            return (ViewLevel.FULL,)
        return _available(runtime)
