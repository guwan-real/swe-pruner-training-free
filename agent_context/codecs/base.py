from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from agent_context.estimation import TokenEstimator
from agent_context.models import (
    ContextSignal,
    ContextView,
    EvidenceDocument,
    EvidenceUnit,
    Observation,
    ObservationKind,
    ViewLevel,
    omitted_ranges,
)
from agent_context.signals import SignalMatch


@dataclass(frozen=True)
class ViewGenerationConfig:
    # Reference-only views require an adapter that exposes ObservationMemoryTools.
    include_reference_view: bool = False
    include_skeleton_view: bool = True
    include_focused_view: bool = True
    focused_expansion_radius: int | None = None
    max_output_chars: int | None = None

    def __post_init__(self) -> None:
        if self.focused_expansion_radius is not None and self.focused_expansion_radius < 0:
            raise ValueError("focused_expansion_radius must be non-negative")
        if self.max_output_chars is not None and self.max_output_chars < 100:
            raise ValueError("max_output_chars must be at least 100")


class ObservationCodec(Protocol):
    name: str
    kind: ObservationKind

    def parse(self, observation: Observation) -> EvidenceDocument: ...

    def generate_views(
        self,
        observation: Observation,
        document: EvidenceDocument,
        signals: Sequence[ContextSignal],
        match: SignalMatch,
        estimator: TokenEstimator,
        config: ViewGenerationConfig,
    ) -> tuple[ContextView, ...]: ...


def render_selected_lines(
    document: EvidenceDocument,
    selected_unit_ids: set[str],
    *,
    level: ViewLevel,
    signal_terms: Sequence[str],
) -> tuple[str, tuple[str, ...], tuple[tuple[int, int], ...]]:
    kept_lines = {
        line_no
        for unit in document.units
        if unit.id in selected_unit_ids
        for line_no in unit.line_numbers
    }
    header = (
        f'<agent_context_view observation_id="{document.observation_id}" '
        f'level="{level.name.lower()}" codec="{document.codec}">\n'
        "This is a reversible view of an older tool observation."
    )
    if signal_terms:
        header += " Matched signals: " + ", ".join(sorted(signal_terms)[:12]) + "."
    header += "\n</agent_context_view>"
    output = [header]
    line_no = 1
    while line_no <= len(document.lines):
        if line_no in kept_lines:
            output.append(document.lines[line_no - 1])
            line_no += 1
            continue
        start = line_no
        while line_no <= len(document.lines) and line_no not in kept_lines:
            line_no += 1
        end = line_no - 1
        label = f"line {start}" if start == end else f"lines {start}-{end}"
        output.append(f"... [agent-context omitted {label}] ...")
    selected = tuple(sorted(selected_unit_ids))
    return "\n".join(output), selected, omitted_ranges(len(document.lines), sorted(kept_lines))


def reference_view(
    observation: Observation,
    document: EvidenceDocument,
    estimator: TokenEstimator,
) -> ContextView:
    path = f' path="{observation.path}"' if observation.path else ""
    content = (
        f'<observation_reference id="{observation.id}" kind="{observation.kind.value}"{path}>\n'
        f"Older tool output archived with {len(document.lines)} visible lines. "
        f"Use memory.search or memory.read with observation id {observation.id} to inspect it.\n"
        "</observation_reference>"
    )
    return ContextView(
        observation_id=observation.id,
        level=ViewLevel.REFERENCE,
        content=content,
        token_count=estimator.estimate(content),
        codec=document.codec,
        policy="reference",
        reason="global-budget-reference",
    )


def full_view(
    observation: Observation,
    document: EvidenceDocument,
    estimator: TokenEstimator,
) -> ContextView:
    return ContextView(
        observation_id=observation.id,
        level=ViewLevel.FULL,
        content=observation.visible_content,
        token_count=estimator.estimate(observation.visible_content),
        codec=document.codec,
        policy="full",
        preserved_unit_ids=tuple(unit.id for unit in document.units),
        reason="full-observation",
    )


def build_compact_view(
    observation: Observation,
    document: EvidenceDocument,
    selected_unit_ids: set[str],
    *,
    level: ViewLevel,
    policy: str,
    reason: str,
    match: SignalMatch,
    estimator: TokenEstimator,
) -> ContextView:
    content, selected, omitted = render_selected_lines(
        document,
        selected_unit_ids,
        level=level,
        signal_terms=tuple(match.matched_terms),
    )
    return ContextView(
        observation_id=observation.id,
        level=level,
        content=content,
        token_count=estimator.estimate(content),
        codec=document.codec,
        policy=policy,
        preserved_unit_ids=selected,
        omitted_line_ranges=omitted,
        relevance_score=sum(match.unit_scores.get(unit_id, 0.0) for unit_id in selected),
        reason=reason,
        metadata={"matched_terms": sorted(match.matched_terms)},
    )


def expand_unit_ids(
    document: EvidenceDocument,
    selected: set[str],
    *,
    radius: int,
) -> set[str]:
    indices = {index for index, unit in enumerate(document.units) if unit.id in selected}
    expanded = set(indices)
    for index in indices:
        expanded.update(range(max(0, index - radius), min(len(document.units), index + radius + 1)))
    return {document.units[index].id for index in expanded}


def mandatory_unit_ids(document: EvidenceDocument) -> set[str]:
    return {unit.id for unit in document.units if unit.mandatory}


def ensure_endpoints(units: Sequence[EvidenceUnit]) -> tuple[EvidenceUnit, ...]:
    if not units:
        return ()
    endpoint_ids = {units[0].id, units[-1].id}
    return tuple(
        EvidenceUnit(
            id=unit.id,
            start_line=unit.start_line,
            end_line=unit.end_line,
            text=unit.text,
            reasons=unit.reasons,
            terms=unit.terms,
            mandatory=unit.mandatory or unit.id in endpoint_ids,
        )
        for unit in units
    )


def filter_views_by_size(
    views: Sequence[ContextView],
    *,
    full: ContextView,
    max_output_chars: int | None,
) -> tuple[ContextView, ...]:
    values: dict[ViewLevel, ContextView] = {ViewLevel.FULL: full}
    for view in views:
        if view.level == ViewLevel.FULL:
            continue
        if len(view.content) >= len(full.content):
            continue
        if max_output_chars is not None and len(view.content) > max_output_chars:
            continue
        previous = values.get(view.level)
        if previous is None or view.token_count < previous.token_count:
            values[view.level] = view
    return tuple(values[level] for level in sorted(values))


class CodecRegistry:
    def __init__(self, codecs: Mapping[ObservationKind, ObservationCodec] | None = None) -> None:
        self._codecs = dict(codecs or {})

    def register(self, codec: ObservationCodec) -> None:
        self._codecs[codec.kind] = codec

    def get(self, kind: ObservationKind) -> ObservationCodec:
        try:
            return self._codecs[kind]
        except KeyError as exc:
            raise KeyError(f"no observation codec registered for {kind.value}") from exc

    def names(self) -> dict[str, str]:
        return {
            kind.value: codec.name
            for kind, codec in sorted(self._codecs.items(), key=lambda x: x[0].value)
        }
