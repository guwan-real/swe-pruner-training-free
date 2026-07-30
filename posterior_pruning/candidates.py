from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.:/-]*|\d+")
ERROR_RE = re.compile(
    r"\b(?:traceback|error|exception|failed|failure|fatal|panic|assertion|warning)\b",
    re.IGNORECASE,
)
DIFF_RE = re.compile(r"^(?:diff --git|index [0-9a-f]+|--- |\+\+\+ |@@ )")
STRUCTURE_RE = re.compile(
    r"^\s*(?:async\s+def|def|class|interface|enum|struct|trait|impl|"
    r"function|func|fn|import|from|package|use|#include)\b"
)
LOCATION_RE = re.compile(r"(?:^|[\s\"'])[\w./-]+\.(?:py|js|ts|tsx|go|rs|java|c|cc|h):\d+")
STOPWORDS = {
    "and",
    "are",
    "bash",
    "for",
    "from",
    "have",
    "into",
    "not",
    "that",
    "the",
    "this",
    "with",
    "your",
}


def estimate_tokens(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text, re.UNICODE))


@dataclass(frozen=True)
class ObservationBlock:
    start_line: int
    end_line: int
    text: str
    utility: float
    protected: bool
    reasons: tuple[str, ...]

    @property
    def line_numbers(self) -> tuple[int, ...]:
        return tuple(range(self.start_line, self.end_line + 1))


@dataclass(frozen=True)
class ObservationCandidate:
    text: str
    keep_ratio: float
    kept_line_numbers: tuple[int, ...]
    selected_block_count: int


@dataclass(frozen=True)
class CandidateConfig:
    block_max_lines: int = 12
    protect_errors: bool = True
    protect_diffs: bool = True
    protect_edge_lines: bool = True

    def __post_init__(self) -> None:
        if self.block_max_lines < 1:
            raise ValueError("block_max_lines must be positive")


def _terms(text: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and not token.isdigit() and token.lower() not in STOPWORDS
    }


def _partition_ranges(lines: Sequence[str], max_lines: int) -> list[tuple[int, int]]:
    if not lines:
        return []
    ranges: list[tuple[int, int]] = []
    start = 1
    for line_no, line in enumerate(lines, start=1):
        reached_limit = line_no - start + 1 >= max_lines
        paragraph_end = not line.strip() and line_no > start
        if reached_limit or paragraph_end:
            ranges.append((start, line_no))
            start = line_no + 1
    if start <= len(lines):
        ranges.append((start, len(lines)))
    # Edge protection should protect the first/last line, not an entire
    # max_lines-sized block. Split those lines into singleton blocks so greedy
    # methods can still operate on short and medium observations.
    split_ranges: list[tuple[int, int]] = []
    for range_start, range_end in ranges:
        if range_start == 1 and range_end > range_start:
            split_ranges.append((1, 1))
            range_start += 1
        protect_tail = range_end == len(lines) and range_end > range_start
        if protect_tail:
            range_end -= 1
        if range_start <= range_end:
            split_ranges.append((range_start, range_end))
        if protect_tail:
            split_ranges.append((len(lines), len(lines)))
    return split_ranges


def build_blocks(
    observation: str,
    *,
    next_action: str,
    query: str = "",
    config: CandidateConfig | None = None,
) -> list[ObservationBlock]:
    config = config or CandidateConfig()
    lines = observation.splitlines()
    action_terms = _terms(f"{query}\n{next_action}")
    blocks: list[ObservationBlock] = []
    ranges = _partition_ranges(lines, config.block_max_lines)
    for block_index, (start, end) in enumerate(ranges):
        text = "\n".join(lines[start - 1 : end])
        block_terms = _terms(text)
        overlap = action_terms & block_terms
        reasons: list[str] = []
        utility = float(len(overlap) * 8)
        if overlap:
            reasons.append("action-overlap")
        has_error = bool(ERROR_RE.search(text))
        has_diff = any(DIFF_RE.search(line) for line in lines[start - 1 : end])
        if has_error:
            utility += 100.0
            reasons.append("error")
        if has_diff:
            utility += 90.0
            reasons.append("diff")
        if any(STRUCTURE_RE.search(line) for line in lines[start - 1 : end]):
            utility += 24.0
            reasons.append("structure")
        if LOCATION_RE.search(text):
            utility += 20.0
            reasons.append("source-location")
        edge = block_index == 0 or block_index == len(ranges) - 1
        if edge:
            utility += 6.0
            reasons.append("edge")
        protected = (
            (config.protect_errors and has_error)
            or (config.protect_diffs and has_diff)
            or (config.protect_edge_lines and edge and len(ranges) > 1)
        )
        blocks.append(
            ObservationBlock(
                start_line=start,
                end_line=end,
                text=text,
                utility=utility,
                protected=protected,
                reasons=tuple(reasons),
            )
        )
    return blocks


def render_kept_lines(lines: Sequence[str], kept_lines: Iterable[int]) -> str:
    kept = set(kept_lines)
    if not lines:
        return ""
    output: list[str] = []
    line_no = 1
    while line_no <= len(lines):
        if line_no in kept:
            output.append(lines[line_no - 1])
            line_no += 1
            continue
        omitted_start = line_no
        while line_no <= len(lines) and line_no not in kept:
            line_no += 1
        omitted_end = line_no - 1
        if omitted_start == omitted_end:
            output.append(f"... [posterior-pruned line {omitted_start}] ...")
        else:
            output.append(f"... [posterior-pruned lines {omitted_start}-{omitted_end}] ...")
    return "\n".join(output)


def candidate_for_ratio(
    observation: str,
    *,
    next_action: str,
    keep_ratio: float,
    query: str = "",
    config: CandidateConfig | None = None,
) -> ObservationCandidate:
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be in (0, 1]")
    lines = observation.splitlines()
    if not lines:
        return ObservationCandidate("", keep_ratio, (), 0)
    blocks = build_blocks(
        observation,
        next_action=next_action,
        query=query,
        config=config,
    )
    target_lines = max(1, math.ceil(len(lines) * keep_ratio))
    selected: set[int] = set()
    selected_blocks: set[int] = set()
    for index, block in enumerate(blocks):
        if block.protected:
            selected.update(block.line_numbers)
            selected_blocks.add(index)
    for index, block in sorted(
        enumerate(blocks),
        key=lambda item: (-item[1].utility, item[1].start_line),
    ):
        if len(selected) >= target_lines:
            break
        needed = target_lines - len(selected)
        selected.update(block.line_numbers[:needed])
        selected_blocks.add(index)
    kept = tuple(sorted(selected))
    return ObservationCandidate(
        text=render_kept_lines(lines, kept),
        keep_ratio=len(kept) / len(lines),
        kept_line_numbers=kept,
        selected_block_count=len(selected_blocks),
    )


def deletion_order(blocks: Sequence[ObservationBlock]) -> list[ObservationBlock]:
    return sorted(
        (block for block in blocks if not block.protected),
        key=lambda block: (block.utility, -block.start_line),
    )
