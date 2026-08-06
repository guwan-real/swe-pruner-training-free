from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Callable

from agent_context.codecs.base import (
    CodecRegistry,
    ViewGenerationConfig,
    build_compact_view,
    ensure_endpoints,
    expand_unit_ids,
    filter_views_by_size,
    full_view,
    mandatory_unit_ids,
    reference_view,
)
from agent_context.estimation import TokenEstimator
from agent_context.models import (
    ContextSignal,
    ContextView,
    EvidenceDocument,
    EvidenceUnit,
    Observation,
    ObservationKind,
    ViewLevel,
)
from agent_context.signals import SignalMatch, lexical_terms

LOCATION_RE = re.compile(
    r"(?P<path>[A-Za-z0-9_./-]+\.(?:py|js|jsx|ts|tsx|go|rs|java|c|cc|cpp|h|hpp|rb|php|sh))"
    r"(?::(?P<line>\d+))?"
)
ERROR_RE = re.compile(
    r"\b(?:traceback|error|exception|failed|failure|fatal|panic|assertion|segfault|warning)\b",
    re.IGNORECASE,
)
TRACE_FRAME_RE = re.compile(r'^\s*(?:File\s+"[^"]+",\s+line\s+\d+|at\s+\S+.*:\d+)')
DIFF_RE = re.compile(r"^(?:diff --git|index [0-9a-f]+|--- |\+\+\+ |@@ |\+[^+]|-[^-])")
STRUCTURE_RE = re.compile(
    r"^\s*(?:async\s+def|def|class|interface|enum|struct|trait|impl|function|func|fn|"
    r"import|from|package|use|#include|module|namespace)\b"
)
TEST_RE = re.compile(
    r"(?:=+\s+(?:FAILURES|ERRORS|short test summary)|\b(?:FAILED|ERROR)\b|"
    r"\d+\s+failed\b|AssertionError)",
    re.IGNORECASE,
)
TREE_RE = re.compile(r"^(?:[│├└─ ]{2,}|[.A-Za-z0-9_-]+/)\S*")


def _shell_verb(command: str) -> str:
    try:
        words = shlex.split(command)
    except ValueError:
        words = command.split()
    for word in words:
        name = PurePosixPath(word).name.lower()
        if "=" in word and not word.startswith(("/", "./", "../")):
            continue
        if name in {"env", "sudo", "timeout", "command", "xargs"} or name.startswith("-"):
            continue
        return name
    return ""


def classify_observation(text: str, *, command: str = "", path: str = "") -> ObservationKind:
    lines = text.splitlines()
    command_lower = command.lower()
    verb = _shell_verb(command)
    if "git diff" in command_lower or sum(bool(DIFF_RE.search(line)) for line in lines) >= 3:
        return ObservationKind.DIFF
    if (
        "traceback (most recent call last)" in text.lower()
        or sum(bool(TRACE_FRAME_RE.search(line)) for line in lines) >= 2
    ):
        return ObservationKind.TRACEBACK
    if TEST_RE.search(text) or (
        verb in {"pytest", "tox", "jest", "npm", "pnpm", "yarn", "go", "cargo"}
        and "test" in command_lower
    ):
        return ObservationKind.TEST_LOG
    if verb in {"grep", "rg", "ag", "ack", "find"}:
        return ObservationKind.SEARCH
    if verb in {"tree", "ls"} and sum(bool(TREE_RE.search(line)) for line in lines[:100]) >= 3:
        return ObservationKind.TREE
    source_suffixes = (
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".go",
        ".rs",
        ".java",
        ".c",
        ".cc",
        ".cpp",
        ".h",
        ".hpp",
        ".rb",
        ".php",
        ".sh",
    )
    if path.lower().endswith(source_suffixes):
        return ObservationKind.SOURCE
    if len(lines) >= 20 and sum(bool(STRUCTURE_RE.search(line)) for line in lines) >= max(
        2, len(lines) // 50
    ):
        return ObservationKind.SOURCE
    return ObservationKind.GENERIC


def _reasons(line: str) -> frozenset[str]:
    values: set[str] = set()
    if ERROR_RE.search(line):
        values.add("error")
    if TRACE_FRAME_RE.search(line):
        values.add("trace-frame")
    if DIFF_RE.search(line):
        values.add("diff")
    if STRUCTURE_RE.search(line):
        values.add("structure")
    if LOCATION_RE.search(line):
        values.add("source-location")
    if TEST_RE.search(line):
        values.add("test-failure")
    return frozenset(values)


def _chunk_ranges(start: int, end: int, size: int):
    while start <= end:
        chunk_end = min(end, start + size - 1)
        yield start, chunk_end
        start = chunk_end + 1


def _semantic_units(
    observation: Observation,
    *,
    block_max_lines: int,
    special: Callable[[frozenset[str]], bool],
    mandatory: Callable[[frozenset[str]], bool],
    one_line_units: bool = False,
) -> tuple[EvidenceUnit, ...]:
    lines = observation.visible_content.splitlines()
    if not lines:
        return ()
    if one_line_units:
        ranges = [(index, index) for index in range(1, len(lines) + 1)]
    else:
        special_lines = {
            index: _reasons(line)
            for index, line in enumerate(lines, start=1)
            if special(_reasons(line))
        }
        ranges: list[tuple[int, int]] = []
        cursor = 1
        for line_no in sorted(special_lines):
            if cursor < line_no:
                ranges.extend(_chunk_ranges(cursor, line_no - 1, block_max_lines))
            ranges.append((line_no, line_no))
            cursor = line_no + 1
        if cursor <= len(lines):
            ranges.extend(_chunk_ranges(cursor, len(lines), block_max_lines))
    units: list[EvidenceUnit] = []
    for index, (start, end) in enumerate(ranges):
        text = "\n".join(lines[start - 1 : end])
        reasons = frozenset(reason for line in lines[start - 1 : end] for reason in _reasons(line))
        units.append(
            EvidenceUnit(
                id=f"{observation.id}:u{index}",
                start_line=start,
                end_line=end,
                text=text,
                reasons=reasons,
                terms=lexical_terms(text),
                mandatory=mandatory(reasons),
            )
        )
    return ensure_endpoints(units)


@dataclass(frozen=True)
class TypedLineCodec:
    name: str
    kind: ObservationKind
    block_max_lines: int = 16
    expansion_radius: int = 0
    special_reasons: frozenset[str] = frozenset()
    mandatory_reasons: frozenset[str] = frozenset()
    one_line_units: bool = False
    allow_compaction: bool = True

    @staticmethod
    def _fit_selected_to_cap(
        observation: Observation,
        document: EvidenceDocument,
        selected: set[str],
        mandatory: set[str],
        match: SignalMatch,
        estimator: TokenEstimator,
        max_output_chars: int | None,
    ) -> set[str]:
        if max_output_chars is None:
            return selected
        fitted = set(selected)
        matched = [unit_id for unit_id in match.matched_unit_ids if unit_id in fitted]
        protected = set(mandatory)
        if matched:
            protected.add(max(matched, key=lambda unit_id: match.unit_scores[unit_id]))
        unit_order = {unit.id: index for index, unit in enumerate(document.units)}
        while True:
            candidate = build_compact_view(
                observation,
                document,
                fitted,
                level=ViewLevel.FOCUSED,
                policy="typed-posterior-focus",
                reason="posterior-signal-match",
                match=match,
                estimator=estimator,
            )
            if len(candidate.content) <= max_output_chars:
                return fitted
            removable = fitted.difference(protected)
            if not removable:
                return fitted
            remove = min(
                removable,
                key=lambda unit_id: (
                    match.unit_scores.get(unit_id, 0.0),
                    -unit_order[unit_id],
                ),
            )
            fitted.remove(remove)

    def parse(self, observation: Observation) -> EvidenceDocument:
        units = _semantic_units(
            observation,
            block_max_lines=self.block_max_lines,
            special=lambda reasons: bool(self.special_reasons.intersection(reasons)),
            mandatory=lambda reasons: bool(self.mandatory_reasons.intersection(reasons)),
            one_line_units=self.one_line_units,
        )
        return EvidenceDocument(
            observation_id=observation.id,
            kind=observation.kind,
            lines=tuple(observation.visible_content.splitlines()),
            units=units,
            codec=self.name,
        )

    def generate_views(
        self,
        observation: Observation,
        document: EvidenceDocument,
        signals: tuple[ContextSignal, ...] | list[ContextSignal],
        match: SignalMatch,
        estimator: TokenEstimator,
        config: ViewGenerationConfig,
    ) -> tuple[ContextView, ...]:
        del signals
        full = full_view(observation, document, estimator)
        if not self.allow_compaction or not document.units:
            return (full,)
        values: list[ContextView] = []
        mandatory = mandatory_unit_ids(document)
        if config.include_reference_view:
            values.append(reference_view(observation, document, estimator))
        if config.include_skeleton_view and mandatory:
            values.append(
                build_compact_view(
                    observation,
                    document,
                    mandatory,
                    level=ViewLevel.SKELETON,
                    policy="typed-skeleton",
                    reason="mandatory-evidence",
                    match=match,
                    estimator=estimator,
                )
            )
        matched = set(match.matched_unit_ids)
        if config.include_focused_view and matched:
            radius = (
                self.expansion_radius
                if config.focused_expansion_radius is None
                else config.focused_expansion_radius
            )
            selected = mandatory.union(expand_unit_ids(document, matched, radius=radius))
            selected = self._fit_selected_to_cap(
                observation,
                document,
                selected,
                mandatory,
                match,
                estimator,
                config.max_output_chars,
            )
            values.append(
                build_compact_view(
                    observation,
                    document,
                    selected,
                    level=ViewLevel.FOCUSED,
                    policy="typed-posterior-focus",
                    reason="posterior-signal-match",
                    match=match,
                    estimator=estimator,
                )
            )
        return filter_views_by_size(
            (*values, full),
            full=full,
            max_output_chars=config.max_output_chars,
        )


def build_typed_codec_registry(*, block_max_lines: int = 16) -> CodecRegistry:
    registry = CodecRegistry()
    registry.register(
        TypedLineCodec(
            name="source_structure",
            kind=ObservationKind.SOURCE,
            block_max_lines=block_max_lines,
            expansion_radius=1,
            special_reasons=frozenset({"structure", "error", "source-location"}),
            mandatory_reasons=frozenset({"structure", "error"}),
        )
    )
    registry.register(
        TypedLineCodec(
            name="diff_full",
            kind=ObservationKind.DIFF,
            block_max_lines=block_max_lines,
            allow_compaction=False,
        )
    )
    registry.register(
        TypedLineCodec(
            name="traceback_frames",
            kind=ObservationKind.TRACEBACK,
            block_max_lines=block_max_lines,
            expansion_radius=2,
            special_reasons=frozenset({"error", "trace-frame", "source-location"}),
            mandatory_reasons=frozenset({"error", "trace-frame"}),
        )
    )
    registry.register(
        TypedLineCodec(
            name="test_failures",
            kind=ObservationKind.TEST_LOG,
            block_max_lines=block_max_lines,
            expansion_radius=2,
            special_reasons=frozenset({"error", "trace-frame", "source-location", "test-failure"}),
            mandatory_reasons=frozenset({"error", "trace-frame", "test-failure"}),
        )
    )
    registry.register(
        TypedLineCodec(
            name="search_hits",
            kind=ObservationKind.SEARCH,
            block_max_lines=1,
            one_line_units=True,
            mandatory_reasons=frozenset({"error"}),
        )
    )
    registry.register(
        TypedLineCodec(
            name="tree_paths",
            kind=ObservationKind.TREE,
            block_max_lines=1,
            one_line_units=True,
            mandatory_reasons=frozenset({"error"}),
        )
    )
    registry.register(
        TypedLineCodec(
            name="generic_evidence",
            kind=ObservationKind.GENERIC,
            block_max_lines=block_max_lines,
            expansion_radius=1,
            special_reasons=frozenset({"error", "source-location", "test-failure"}),
            mandatory_reasons=frozenset({"error", "test-failure"}),
        )
    )
    return registry
