from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from types import MappingProxyType
from typing import Any, Mapping, Sequence


def immutable_mapping(values: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    return MappingProxyType(dict(values or {}))


class ObservationKind(str, Enum):
    SOURCE = "source"
    DIFF = "diff"
    TRACEBACK = "traceback"
    TEST_LOG = "test_log"
    SEARCH = "search"
    TREE = "tree"
    GENERIC = "generic"


class LifecycleStage(str, Enum):
    CAPTURED = "captured"
    SEEN = "seen"
    ENRICHED = "enriched"
    ARCHIVED = "archived"


class MemoryTier(str, Enum):
    HOT = "hot"
    COLD = "cold"
    PINNED = "pinned"


class ViewLevel(IntEnum):
    REFERENCE = 0
    SKELETON = 1
    FOCUSED = 2
    FULL = 3

    @classmethod
    def parse(cls, value: str | int | ViewLevel) -> ViewLevel:
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            return cls(value)
        return cls[value.strip().upper()]


@dataclass(frozen=True)
class Observation:
    id: str
    task_id: str
    step: int
    raw_content: str
    visible_content: str
    causing_action: str = ""
    path: str = ""
    kind: ObservationKind = ObservationKind.GENERIC
    metadata: Mapping[str, Any] = field(default_factory=immutable_mapping)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("observation id must not be empty")
        if self.step < 0:
            raise ValueError("observation step must be non-negative")
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))


@dataclass(frozen=True)
class ActionEvent:
    step: int
    command: str = ""
    context_focus_question: str = ""
    response_content: str = ""
    metadata: Mapping[str, Any] = field(default_factory=immutable_mapping)

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("action step must be non-negative")
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))


@dataclass(frozen=True)
class ContextSignal:
    provider: str
    text: str
    step: int
    weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=immutable_mapping)

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("signal provider must not be empty")
        if self.step < 0:
            raise ValueError("signal step must be non-negative")
        if self.weight < 0:
            raise ValueError("signal weight must be non-negative")
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))


@dataclass(frozen=True)
class EvidenceUnit:
    id: str
    start_line: int
    end_line: int
    text: str
    reasons: frozenset[str] = frozenset()
    terms: frozenset[str] = frozenset()
    mandatory: bool = False

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("evidence unit id must not be empty")
        if self.start_line < 1 or self.end_line < self.start_line:
            raise ValueError("evidence unit line range is invalid")

    @property
    def line_numbers(self) -> tuple[int, ...]:
        return tuple(range(self.start_line, self.end_line + 1))


@dataclass(frozen=True)
class EvidenceDocument:
    observation_id: str
    kind: ObservationKind
    lines: tuple[str, ...]
    units: tuple[EvidenceUnit, ...]
    codec: str
    metadata: Mapping[str, Any] = field(default_factory=immutable_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))

    @property
    def mandatory_unit_ids(self) -> frozenset[str]:
        return frozenset(unit.id for unit in self.units if unit.mandatory)


@dataclass(frozen=True)
class ContextView:
    observation_id: str
    level: ViewLevel
    content: str
    token_count: int
    codec: str
    policy: str
    preserved_unit_ids: tuple[str, ...] = ()
    omitted_line_ranges: tuple[tuple[int, int], ...] = ()
    relevance_score: float = 0.0
    reversible: bool = True
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=immutable_mapping)

    def __post_init__(self) -> None:
        if self.token_count < 0:
            raise ValueError("view token_count must be non-negative")
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))


@dataclass
class ObservationRuntime:
    observation: Observation
    document: EvidenceDocument
    stage: LifecycleStage = LifecycleStage.CAPTURED
    tier: MemoryTier = MemoryTier.HOT
    signals: list[ContextSignal] = field(default_factory=list)
    views: dict[ViewLevel, ContextView] = field(default_factory=dict)
    prompt_uses: int = 0
    compacted_prompt_uses: int = 0
    estimated_tokens_saved: int = 0
    last_referenced_step: int | None = None
    pinned: bool = False
    committed_view: ContextView | None = None
    committed_prompt_index: int | None = None
    last_rendered_view_fingerprint: str | None = None
    view_switch_count: int = 0

    def add_view(self, view: ContextView) -> None:
        if view.observation_id != self.observation.id:
            raise ValueError("view belongs to a different observation")
        self.views[view.level] = view

    def commit_view(self, view: ContextView, *, prompt_index: int) -> None:
        if view.observation_id != self.observation.id:
            raise ValueError("cannot commit a view from a different observation")
        if prompt_index < 1:
            raise ValueError("committed prompt index must be positive")
        if self.committed_view is not None:
            if self.committed_view != view:
                raise RuntimeError(f"observation {self.observation.id} view is already committed")
            return
        self.committed_view = view
        self.committed_prompt_index = prompt_index

    @property
    def full_view(self) -> ContextView:
        try:
            return self.views[ViewLevel.FULL]
        except KeyError as exc:
            raise RuntimeError(f"observation {self.observation.id} has no full view") from exc


@dataclass(frozen=True)
class PromptManifestEntry:
    observation_id: str
    message_index: int
    kind: str
    stage: str
    tier: str
    selected_level: str
    full_tokens: int
    selected_tokens: int
    saved_tokens: int
    codec: str
    policy: str
    reason: str
    committed: bool
    committed_prompt_index: int | None
    selection_changed: bool
    view_fingerprint: str
    preserved_unit_ids: tuple[str, ...] = ()
    omitted_line_ranges: tuple[tuple[int, int], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "message_index": self.message_index,
            "kind": self.kind,
            "stage": self.stage,
            "tier": self.tier,
            "selected_level": self.selected_level,
            "full_tokens": self.full_tokens,
            "selected_tokens": self.selected_tokens,
            "saved_tokens": self.saved_tokens,
            "codec": self.codec,
            "policy": self.policy,
            "reason": self.reason,
            "committed": self.committed,
            "committed_prompt_index": self.committed_prompt_index,
            "selection_changed": self.selection_changed,
            "view_fingerprint": self.view_fingerprint,
            "preserved_unit_ids": list(self.preserved_unit_ids),
            "omitted_line_ranges": [list(value) for value in self.omitted_line_ranges],
        }


@dataclass(frozen=True)
class PromptManifest:
    task_id: str
    prompt_index: int
    planner: str
    timing: str
    observation_budget: int
    full_observation_tokens: int
    selected_observation_tokens: int
    budget_overflow_tokens: int
    context_view_switches: int
    earliest_context_change_message_index: int | None
    entries: tuple[PromptManifestEntry, ...]

    @property
    def estimated_tokens_saved(self) -> int:
        return max(0, self.full_observation_tokens - self.selected_observation_tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prompt_index": self.prompt_index,
            "planner": self.planner,
            "timing": self.timing,
            "observation_budget": self.observation_budget,
            "full_observation_tokens": self.full_observation_tokens,
            "selected_observation_tokens": self.selected_observation_tokens,
            "estimated_tokens_saved": self.estimated_tokens_saved,
            "budget_overflow_tokens": self.budget_overflow_tokens,
            "context_view_switches": self.context_view_switches,
            "earliest_context_change_message_index": self.earliest_context_change_message_index,
            "entries": [entry.to_dict() for entry in self.entries],
        }


@dataclass(frozen=True)
class PromptBuild:
    messages: tuple[dict[str, Any], ...]
    manifest: PromptManifest

    def as_list(self) -> list[dict[str, Any]]:
        return list(self.messages)


def omitted_ranges(line_count: int, kept_lines: Sequence[int]) -> tuple[tuple[int, int], ...]:
    kept = set(kept_lines)
    ranges: list[tuple[int, int]] = []
    line_no = 1
    while line_no <= line_count:
        if line_no in kept:
            line_no += 1
            continue
        start = line_no
        while line_no <= line_count and line_no not in kept:
            line_no += 1
        ranges.append((start, line_no - 1))
    return tuple(ranges)
