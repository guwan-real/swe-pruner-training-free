from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from agent_context.config import PlannerConfig
from agent_context.models import ContextView, ObservationRuntime, ViewLevel
from agent_context.visibility import VisibilityPolicy


@dataclass(frozen=True)
class PlanningRecord:
    runtime: ObservationRuntime
    message_index: int
    recency_rank: int


@dataclass(frozen=True)
class PlanResult:
    selections: Mapping[str, ContextView]
    observation_budget: int
    full_observation_tokens: int
    selected_observation_tokens: int
    budget_overflow_tokens: int


class PromptPlanner(Protocol):
    name: str

    def plan(
        self,
        records: Sequence[PlanningRecord],
        *,
        visibility: VisibilityPolicy,
        non_observation_tokens: int = 0,
    ) -> PlanResult: ...


def _efficient_candidates(
    runtime: ObservationRuntime,
    allowed: Sequence[ViewLevel],
) -> tuple[ContextView, ...]:
    if runtime.committed_view is not None:
        return (runtime.committed_view,)
    values = [runtime.views[level] for level in allowed if level in runtime.views]
    if not values:
        return (runtime.full_view,)
    ordered = sorted(values, key=lambda view: view.level)
    efficient: list[ContextView] = []
    for view in ordered:
        if efficient and view.token_count <= efficient[-1].token_count:
            efficient[-1] = view
        else:
            efficient.append(view)
    if efficient[-1].level != ViewLevel.FULL:
        efficient.append(runtime.full_view)
    return tuple(efficient)


class GlobalBudgetPlanner:
    def __init__(self, config: PlannerConfig) -> None:
        self.config = config
        self.name = (
            "cache_aware_global_budget_greedy_v1"
            if config.cache_policy == "freeze_on_cold"
            else "global_budget_greedy_v1"
        )

    def _budget(self, full_tokens: int, non_observation_tokens: int) -> int:
        if self.config.mode == "passthrough":
            return full_tokens
        if self.config.mode == "retention":
            return round(full_tokens * self.config.target_retention)
        if self.config.mode == "fixed":
            assert self.config.observation_budget is not None
            return self.config.observation_budget
        assert self.config.max_prompt_tokens is not None
        return max(
            0,
            self.config.max_prompt_tokens
            - self.config.reserve_completion_tokens
            - non_observation_tokens,
        )

    def _importance(self, record: PlanningRecord) -> float:
        relevance = max(
            (view.relevance_score for view in record.runtime.views.values()),
            default=0.0,
        )
        recency = 1.0 / (record.recency_rank + 1)
        kind_weight = self.config.kind_weights.get(record.runtime.observation.kind.value, 1.0)
        return kind_weight * (
            1.0
            + self.config.relevance_weight * min(relevance, 10.0)
            + self.config.recency_weight * recency
        )

    def plan(
        self,
        records: Sequence[PlanningRecord],
        *,
        visibility: VisibilityPolicy,
        non_observation_tokens: int = 0,
    ) -> PlanResult:
        full_tokens = sum(record.runtime.full_view.token_count for record in records)
        budget = self._budget(full_tokens, non_observation_tokens)
        if not records:
            return PlanResult({}, budget, 0, 0, 0)

        candidates = {
            record.runtime.observation.id: _efficient_candidates(
                record.runtime,
                visibility.allowed_views(record.runtime),
            )
            for record in records
        }
        positions = {observation_id: 0 for observation_id in candidates}
        selections = {observation_id: values[0] for observation_id, values in candidates.items()}
        selected_tokens = sum(view.token_count for view in selections.values())

        while True:
            upgrades: list[tuple[float, int, str, ContextView]] = []
            records_by_id = {record.runtime.observation.id: record for record in records}
            for observation_id, values in candidates.items():
                position = positions[observation_id]
                if position + 1 >= len(values):
                    continue
                current = values[position]
                next_view = values[position + 1]
                cost = next_view.token_count - current.token_count
                if cost <= 0:
                    positions[observation_id] += 1
                    selections[observation_id] = next_view
                    continue
                importance = self._importance(records_by_id[observation_id])
                utility_gain = max(1, int(next_view.level) - int(current.level))
                score = importance * utility_gain / cost
                upgrades.append(
                    (score, -records_by_id[observation_id].recency_rank, observation_id, next_view)
                )
            if not upgrades:
                break
            upgrades.sort(reverse=True)
            chosen = next(
                (
                    item
                    for item in upgrades
                    if selected_tokens + item[3].token_count - selections[item[2]].token_count
                    <= budget
                ),
                None,
            )
            if chosen is None:
                break
            _, _, observation_id, next_view = chosen
            selected_tokens += next_view.token_count - selections[observation_id].token_count
            positions[observation_id] += 1
            selections[observation_id] = next_view

        return PlanResult(
            selections=selections,
            observation_budget=budget,
            full_observation_tokens=full_tokens,
            selected_observation_tokens=selected_tokens,
            budget_overflow_tokens=max(0, selected_tokens - budget),
        )


class FullPromptPlanner:
    name = "full"

    def plan(
        self,
        records: Sequence[PlanningRecord],
        *,
        visibility: VisibilityPolicy,
        non_observation_tokens: int = 0,
    ) -> PlanResult:
        del visibility, non_observation_tokens
        selections = {record.runtime.observation.id: record.runtime.full_view for record in records}
        total = sum(view.token_count for view in selections.values())
        return PlanResult(selections, total, total, total, 0)


class MinimumViewPlanner:
    """Select every policy-allowed compact candidate for legacy parity arms."""

    name = "minimum_allowed_view_v1"

    def plan(
        self,
        records: Sequence[PlanningRecord],
        *,
        visibility: VisibilityPolicy,
        non_observation_tokens: int = 0,
    ) -> PlanResult:
        del non_observation_tokens
        selections: dict[str, ContextView] = {}
        for record in records:
            runtime = record.runtime
            candidates = _efficient_candidates(runtime, visibility.allowed_views(runtime))
            selections[runtime.observation.id] = min(
                candidates,
                key=lambda view: (view.token_count, int(view.level)),
            )
        full_tokens = sum(record.runtime.full_view.token_count for record in records)
        selected_tokens = sum(view.token_count for view in selections.values())
        return PlanResult(
            selections=selections,
            observation_budget=selected_tokens,
            full_observation_tokens=full_tokens,
            selected_observation_tokens=selected_tokens,
            budget_overflow_tokens=0,
        )
