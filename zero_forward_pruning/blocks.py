from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from zero_forward_pruning.text import (
    OutputKind,
    identifiers,
    line_reasons,
    terms,
)


@dataclass(frozen=True)
class EvidenceBlock:
    index: int
    start_line: int
    end_line: int
    text: str
    reasons: tuple[str, ...]
    terms: tuple[str, ...]
    identifiers: frozenset[str]

    @property
    def line_numbers(self) -> tuple[int, ...]:
        return tuple(range(self.start_line, self.end_line + 1))

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1


def _chunk_ranges(start: int, end: int, max_lines: int) -> Iterable[tuple[int, int]]:
    while start <= end:
        chunk_end = min(end, start + max_lines - 1)
        yield start, chunk_end
        start = chunk_end + 1


def _special_reasons(kind: OutputKind, reasons: tuple[str, ...]) -> tuple[str, ...]:
    if kind == OutputKind.SOURCE:
        allowed = {"structure", "error", "source-location", "diff"}
    elif kind in {OutputKind.TRACEBACK, OutputKind.TEST_LOG}:
        allowed = {"error", "trace-frame", "source-location", "test-failure"}
    elif kind == OutputKind.DIFF:
        allowed = {"diff", "error", "source-location"}
    else:
        allowed = {"error", "source-location", "test-failure"}
    return tuple(reason for reason in reasons if reason in allowed)


def build_blocks(
    text: str,
    *,
    kind: OutputKind,
    max_lines: int = 16,
) -> list[EvidenceBlock]:
    """Split output into stable evidence blocks.

    Semantically important lines become singleton blocks.  This lets the
    renderer preserve a source skeleton without dragging an arbitrary 16-line
    body along with every function signature.
    """

    if max_lines < 1:
        raise ValueError("max_lines must be positive")
    lines = text.splitlines()
    if not lines:
        return []
    special: dict[int, tuple[str, ...]] = {}
    for line_no, line in enumerate(lines, start=1):
        reasons = _special_reasons(kind, line_reasons(line))
        if reasons:
            special[line_no] = reasons
    ranges: list[tuple[int, int, tuple[str, ...]]] = []
    cursor = 1
    for line_no in sorted(special):
        if cursor < line_no:
            ranges.extend(
                (start, end, ()) for start, end in _chunk_ranges(cursor, line_no - 1, max_lines)
            )
        ranges.append((line_no, line_no, special[line_no]))
        cursor = line_no + 1
    if cursor <= len(lines):
        ranges.extend(
            (start, end, ()) for start, end in _chunk_ranges(cursor, len(lines), max_lines)
        )
    blocks: list[EvidenceBlock] = []
    for index, (start, end, reasons) in enumerate(ranges):
        block_text = "\n".join(lines[start - 1 : end])
        blocks.append(
            EvidenceBlock(
                index=index,
                start_line=start,
                end_line=end,
                text=block_text,
                reasons=reasons,
                terms=terms(block_text),
                identifiers=frozenset(identifiers(block_text)),
            )
        )
    return blocks


def hard_block_indices(blocks: Sequence[EvidenceBlock], kind: OutputKind) -> set[int]:
    hard_reasons = {
        OutputKind.SOURCE: {"structure", "error", "source-location"},
        OutputKind.DIFF: {"diff", "error", "source-location"},
        OutputKind.TRACEBACK: {"error", "trace-frame", "source-location"},
        OutputKind.TEST_LOG: {"error", "trace-frame", "source-location", "test-failure"},
        OutputKind.SEARCH: {"error", "source-location"},
        OutputKind.TREE: {"error"},
        OutputKind.GENERIC: {"error", "source-location", "test-failure"},
    }[kind]
    selected = {block.index for block in blocks if hard_reasons.intersection(block.reasons)}
    if blocks:
        selected.add(0)
        selected.add(len(blocks) - 1)
    return selected


def expand_indices(
    selected: Iterable[int],
    *,
    block_count: int,
    radius: int,
) -> set[int]:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    expanded: set[int] = set()
    for index in selected:
        expanded.update(range(max(0, index - radius), min(block_count, index + radius + 1)))
    return expanded


def selected_lines(blocks: Sequence[EvidenceBlock], selected: Iterable[int]) -> tuple[int, ...]:
    indices = set(selected)
    values = {
        line_no for block in blocks if block.index in indices for line_no in block.line_numbers
    }
    return tuple(sorted(values))


def document_frequency(blocks: Sequence[EvidenceBlock]) -> Counter[str]:
    frequency: Counter[str] = Counter()
    for block in blocks:
        frequency.update(set(block.terms))
    return frequency
