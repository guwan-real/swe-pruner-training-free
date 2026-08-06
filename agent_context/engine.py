from __future__ import annotations

import copy
import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from agent_context.config import AgentContextConfig
from agent_context.estimation import DEFAULT_ESTIMATOR, TokenEstimator
from agent_context.models import (
    ActionEvent,
    ContextSignal,
    ContextView,
    LifecycleStage,
    MemoryTier,
    Observation,
    ObservationKind,
    ObservationRuntime,
    PromptBuild,
    PromptManifest,
    PromptManifestEntry,
    ViewLevel,
)
from agent_context.planner import PlanningRecord
from agent_context.registry import (
    DEFAULT_COMPONENT_REGISTRY,
    ComponentRegistry,
    ContextComponents,
)
from agent_context.store import InMemoryObservationStore, ObservationStore


class ObservationBoundaryError(ValueError):
    pass


@dataclass(frozen=True)
class MessageBinding:
    message: Mapping[str, Any]
    observation_id: str
    prefix: str
    suffix: str


class ContextEngine:
    """Model-independent observation memory and prompt materialization runtime."""

    def __init__(
        self,
        config: AgentContextConfig | Mapping[str, Any] | None = None,
        *,
        components: ContextComponents | None = None,
        component_registry: ComponentRegistry = DEFAULT_COMPONENT_REGISTRY,
        estimator: TokenEstimator = DEFAULT_ESTIMATOR,
        store: ObservationStore | None = None,
    ) -> None:
        self.config = (
            config
            if isinstance(config, AgentContextConfig)
            else AgentContextConfig.from_mapping(config)
        )
        self.components = components or component_registry.components(self.config)
        self.estimator = estimator
        self.store = store or InMemoryObservationStore()
        self.task_id = ""
        self.task_text = ""
        self.step = 0
        self.prompt_index = 0
        self._bindings: dict[int, MessageBinding] = {}
        self._pending_followups: list[str] = []
        self._last_prompt_observation_ids: tuple[str, ...] = ()
        self._manifests: list[PromptManifest] = []

    def start_task(self, task_id: str, *, task_text: str = "") -> None:
        if not task_id:
            raise ValueError("task_id must not be empty")
        self.store.clear()
        self.task_id = task_id
        self.task_text = task_text
        self.step = 0
        self.prompt_index = 0
        self._bindings.clear()
        self._pending_followups.clear()
        self._last_prompt_observation_ids = ()
        self._manifests.clear()

    def _require_task(self) -> None:
        if not self.task_id:
            self.start_task("task")

    def _next_observation_id(self) -> str:
        return f"obs-{len(self.store.values()) + 1:04d}"

    @staticmethod
    def _resolve_boundary(
        message: Mapping[str, Any],
        visible_content: str,
        prefix: str | None,
        suffix: str | None,
        boundary_is_validated: bool,
    ) -> tuple[str, str]:
        message_content = message.get("content")
        if not isinstance(message_content, str):
            raise ObservationBoundaryError("observation message content must be a string")
        if prefix is not None or suffix is not None:
            resolved_prefix = prefix or ""
            resolved_suffix = suffix or ""
            if (
                not boundary_is_validated
                and resolved_prefix + visible_content + resolved_suffix != message_content
            ):
                raise ObservationBoundaryError(
                    "prefix + visible_content + suffix does not reconstruct the canonical message"
                )
            return resolved_prefix, resolved_suffix
        position = message_content.find(visible_content)
        if position < 0:
            raise ObservationBoundaryError(
                "visible observation is not present in canonical message"
            )
        end = position + len(visible_content)
        return message_content[:position], message_content[end:]

    def ingest_observation(
        self,
        message: Mapping[str, Any],
        *,
        visible_content: str,
        raw_content: str | None = None,
        causing_action: str = "",
        path: str = "",
        kind: ObservationKind | str | None = None,
        prefix: str | None = None,
        suffix: str | None = None,
        observation_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        initial_signals: Sequence[ContextSignal] = (),
        boundary_is_validated: bool = False,
    ) -> Observation:
        self._require_task()
        if id(message) in self._bindings:
            raise ValueError("canonical message is already bound to an observation")
        resolved_prefix, resolved_suffix = self._resolve_boundary(
            message,
            visible_content,
            prefix,
            suffix,
            boundary_is_validated,
        )
        self.step += 1
        if kind is None:
            from agent_context.codecs import classify_observation

            resolved_kind = classify_observation(
                visible_content,
                command=causing_action,
                path=path,
            )
        elif isinstance(kind, ObservationKind):
            resolved_kind = kind
        else:
            resolved_kind = ObservationKind(str(kind))
        observation = Observation(
            id=observation_id or self._next_observation_id(),
            task_id=self.task_id,
            step=self.step,
            raw_content=visible_content if raw_content is None else raw_content,
            visible_content=visible_content,
            causing_action=causing_action,
            path=path,
            kind=resolved_kind,
            metadata=metadata or {},
        )
        codec = self.components.codecs.get(resolved_kind)
        runtime = ObservationRuntime(
            observation=observation,
            document=codec.parse(observation),
        )
        runtime.signals.extend(initial_signals)
        if self.config.include_task_signal and self.task_text:
            runtime.signals.append(
                ContextSignal(provider="task", text=self.task_text, step=self.step, weight=1.0)
            )
        if self.config.include_causing_action_signal and causing_action:
            runtime.signals.append(
                ContextSignal(
                    provider="causing_action",
                    text=causing_action,
                    step=self.step,
                    weight=1.0,
                )
            )
        self._refresh_views(runtime)
        self.store.add(runtime)
        self._bindings[id(message)] = MessageBinding(
            message=message,
            observation_id=observation.id,
            prefix=resolved_prefix,
            suffix=resolved_suffix,
        )
        self._pending_followups.append(observation.id)
        self._refresh_tiers()
        return observation

    def _refresh_views(self, runtime: ObservationRuntime) -> None:
        codec = self.components.codecs.get(runtime.observation.kind)
        match = self.components.signal_strategy.score(runtime.document, runtime.signals)
        views = codec.generate_views(
            runtime.observation,
            runtime.document,
            runtime.signals,
            match,
            self.estimator,
            self.config.views_for_kind(runtime.observation.kind.value),
        )
        runtime.views.clear()
        for view in views:
            runtime.add_view(view)
        if ViewLevel.FULL not in runtime.views:
            raise RuntimeError(f"codec {codec.name} did not produce a full view")

    def _refresh_tiers(self) -> None:
        active = [
            runtime for runtime in self.store.values() if runtime.stage != LifecycleStage.ARCHIVED
        ]
        hot_start = max(0, len(active) - self.config.hot_observations)
        for index, runtime in enumerate(active):
            if runtime.pinned:
                runtime.tier = MemoryTier.PINNED
            elif index >= hot_start and self.config.hot_observations > 0:
                runtime.tier = MemoryTier.HOT
            else:
                runtime.tier = MemoryTier.COLD

    def observe_action(self, event: ActionEvent) -> str | None:
        """Record a normal model response without importing or invoking the model."""

        self.step = max(self.step, event.step)
        for observation_id in self._last_prompt_observation_ids:
            runtime = self.store.get(observation_id)
            if runtime.stage == LifecycleStage.CAPTURED:
                runtime.stage = LifecycleStage.SEEN
        signals = self.components.signal_provider.from_action(event)
        target_id = next(
            (
                observation_id
                for observation_id in reversed(self._pending_followups)
                if observation_id in self._last_prompt_observation_ids
            ),
            None,
        )
        if target_id is not None:
            target = self.store.get(target_id)
            target.signals.extend(signals)
            target.stage = LifecycleStage.ENRICHED
            target.last_referenced_step = event.step
            self._refresh_views(target)
            self._pending_followups.remove(target_id)

        if self.config.track_later_references and signals:
            for runtime in self.store.values():
                if runtime.observation.id == target_id or runtime.stage != LifecycleStage.ENRICHED:
                    continue
                if runtime.committed_view is not None:
                    continue
                match = self.components.signal_strategy.score(runtime.document, signals)
                if not match.matched_unit_ids:
                    continue
                runtime.signals.extend(signals)
                runtime.last_referenced_step = event.step
                self._refresh_views(runtime)
        self._refresh_tiers()
        return target_id

    def _commit_cold_selections(
        self,
        records: Sequence[PlanningRecord],
        selections: Mapping[str, ContextView],
    ) -> None:
        if self.config.planner.cache_policy != "freeze_on_cold":
            return
        for record in records:
            runtime = record.runtime
            if (
                runtime.committed_view is None
                and runtime.stage == LifecycleStage.ENRICHED
                and runtime.tier == MemoryTier.COLD
            ):
                runtime.commit_view(
                    selections[runtime.observation.id],
                    prompt_index=self.prompt_index,
                )

    @staticmethod
    def _view_fingerprint(view: ContextView) -> str:
        payload = f"{view.level.name}\0{view.codec}\0{view.policy}\0{view.content}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _visible_records(
        self, messages: Sequence[Mapping[str, Any]]
    ) -> tuple[tuple[PlanningRecord, ...], dict[int, MessageBinding]]:
        bindings: dict[int, MessageBinding] = {}
        ordered: list[tuple[int, MessageBinding]] = []
        for message_index, message in enumerate(messages):
            binding = self._bindings.get(id(message))
            if binding is None:
                continue
            bindings[message_index] = binding
            ordered.append((message_index, binding))
        records = tuple(
            PlanningRecord(
                runtime=self.store.get(binding.observation_id),
                message_index=message_index,
                recency_rank=len(ordered) - index - 1,
            )
            for index, (message_index, binding) in enumerate(ordered)
        )
        return records, bindings

    def _non_observation_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
        bindings: Mapping[int, MessageBinding],
    ) -> int:
        total = 0
        for index, message in enumerate(messages):
            binding = bindings.get(index)
            if binding is not None:
                total += self.estimator.estimate(binding.prefix + binding.suffix)
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                total += self.estimator.estimate(content)
        return total

    def build_prompt(self, messages: Sequence[Mapping[str, Any]]) -> PromptBuild:
        self._require_task()
        self.prompt_index += 1
        self._refresh_tiers()
        records, bindings = self._visible_records(messages)
        plan = self.components.planner.plan(
            records,
            visibility=self.components.visibility,
            non_observation_tokens=self._non_observation_tokens(messages, bindings),
        )
        self._commit_cold_selections(records, plan.selections)
        rendered: list[dict[str, Any]] = []
        manifest_entries: list[PromptManifestEntry] = []
        changed_message_indices: list[int] = []
        records_by_index = {record.message_index: record for record in records}
        for message_index, message in enumerate(messages):
            view_message = copy.deepcopy(dict(message))
            for field in self.config.telemetry_fields:
                view_message.pop(field, None)
            binding = bindings.get(message_index)
            if binding is not None:
                runtime = self.store.get(binding.observation_id)
                selected = plan.selections[binding.observation_id]
                view_message["content"] = binding.prefix + selected.content + binding.suffix
                full = runtime.full_view
                saved = max(0, full.token_count - selected.token_count)
                fingerprint = self._view_fingerprint(selected)
                selection_changed = (
                    runtime.last_rendered_view_fingerprint is not None
                    and runtime.last_rendered_view_fingerprint != fingerprint
                )
                if selection_changed:
                    runtime.view_switch_count += 1
                    changed_message_indices.append(message_index)
                runtime.last_rendered_view_fingerprint = fingerprint
                runtime.prompt_uses += 1
                if selected.level != ViewLevel.FULL:
                    runtime.compacted_prompt_uses += 1
                    runtime.estimated_tokens_saved += saved
                record = records_by_index[message_index]
                manifest_entries.append(
                    PromptManifestEntry(
                        observation_id=runtime.observation.id,
                        message_index=message_index,
                        kind=runtime.observation.kind.value,
                        stage=runtime.stage.value,
                        tier=runtime.tier.value,
                        selected_level=selected.level.name.lower(),
                        full_tokens=full.token_count,
                        selected_tokens=selected.token_count,
                        saved_tokens=saved,
                        codec=selected.codec,
                        policy=selected.policy,
                        reason=selected.reason,
                        committed=runtime.committed_view is not None,
                        committed_prompt_index=runtime.committed_prompt_index,
                        selection_changed=selection_changed,
                        view_fingerprint=fingerprint,
                        preserved_unit_ids=selected.preserved_unit_ids,
                        omitted_line_ranges=selected.omitted_line_ranges,
                    )
                )
                del record
            rendered.append(view_message)
        manifest = PromptManifest(
            task_id=self.task_id,
            prompt_index=self.prompt_index,
            planner=self.components.planner.name,
            timing=self.components.visibility.name,
            observation_budget=plan.observation_budget,
            full_observation_tokens=plan.full_observation_tokens,
            selected_observation_tokens=plan.selected_observation_tokens,
            budget_overflow_tokens=plan.budget_overflow_tokens,
            context_view_switches=len(changed_message_indices),
            earliest_context_change_message_index=(
                min(changed_message_indices) if changed_message_indices else None
            ),
            entries=tuple(manifest_entries),
        )
        self._last_prompt_observation_ids = tuple(
            record.runtime.observation.id for record in records
        )
        self._manifests.append(manifest)
        return PromptBuild(messages=tuple(rendered), manifest=manifest)

    def pin(self, observation_id: str) -> None:
        runtime = self.store.get(observation_id)
        runtime.pinned = True
        runtime.tier = MemoryTier.PINNED

    def unpin(self, observation_id: str) -> None:
        runtime = self.store.get(observation_id)
        runtime.pinned = False
        self._refresh_tiers()

    def _record_retrieval(self, runtime: ObservationRuntime) -> None:
        # Retrieval content is returned to the caller and must be appended at
        # the prompt tail. Rewriting an older slot would invalidate all KV
        # blocks after that observation.
        runtime.last_referenced_step = self.step

    def read(
        self,
        observation_id: str,
        *,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> str:
        runtime = self.store.get(observation_id)
        lines = runtime.observation.raw_content.splitlines()
        if start_line < 1:
            raise ValueError("start_line must be at least 1")
        resolved_end = len(lines) if end_line is None else end_line
        if resolved_end < start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        self._record_retrieval(runtime)
        return "\n".join(lines[start_line - 1 : resolved_end])

    def search(
        self,
        observation_id: str,
        pattern: str,
        *,
        regex: bool = False,
        max_results: int = 50,
    ) -> tuple[tuple[int, str], ...]:
        if not pattern:
            raise ValueError("search pattern must not be empty")
        if max_results < 1:
            raise ValueError("max_results must be positive")
        runtime = self.store.get(observation_id)
        matcher = re.compile(pattern) if regex else None
        values: list[tuple[int, str]] = []
        for line_no, line in enumerate(runtime.observation.raw_content.splitlines(), start=1):
            matches = bool(matcher.search(line)) if matcher is not None else pattern in line
            if matches:
                values.append((line_no, line))
                if len(values) >= max_results:
                    break
        self._record_retrieval(runtime)
        return tuple(values)

    def finish_task(self) -> dict[str, Any]:
        report = self.report()
        for runtime in self.store.values():
            runtime.stage = LifecycleStage.ARCHIVED
        return report

    @property
    def manifests(self) -> tuple[PromptManifest, ...]:
        return tuple(self._manifests)

    def report(self) -> dict[str, Any]:
        runtimes = self.store.values()
        kinds = Counter(runtime.observation.kind.value for runtime in runtimes)
        stages = Counter(runtime.stage.value for runtime in runtimes)
        tiers = Counter(runtime.tier.value for runtime in runtimes)
        return {
            "version": 1,
            "task_id": self.task_id,
            "timing": self.components.visibility.name,
            "planner": self.components.planner.name,
            "codec_profile": self.config.codec_profile,
            "signal_provider": self.components.signal_provider.name,
            "signal_strategy": self.components.signal_strategy.name,
            "model_forward_count": 0,
            "llm_token_count": 0,
            "observations": len(runtimes),
            "observation_kinds": dict(sorted(kinds.items())),
            "lifecycle_stages": dict(sorted(stages.items())),
            "memory_tiers": dict(sorted(tiers.items())),
            "prompt_views": len(self._manifests),
            "compacted_prompt_uses": sum(runtime.compacted_prompt_uses for runtime in runtimes),
            "estimated_tokens_saved": sum(runtime.estimated_tokens_saved for runtime in runtimes),
            "pinned_observations": sum(runtime.pinned for runtime in runtimes),
            "committed_observations": sum(
                runtime.committed_view is not None for runtime in runtimes
            ),
            "context_view_switches": sum(runtime.view_switch_count for runtime in runtimes),
            "component_manifest": {
                "codecs": self.components.codecs.names(),
                "visibility": self.components.visibility.name,
                "planner": self.components.planner.name,
            },
        }
