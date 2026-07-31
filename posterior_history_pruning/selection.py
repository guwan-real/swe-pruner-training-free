from __future__ import annotations

import re
import shlex
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from math import ceil
from pathlib import PurePosixPath
from typing import Iterable, Sequence

from posterior_history_pruning.protocol import (
    CompactionResult,
    PosteriorHistoryConfig,
    PosteriorSignal,
)

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]*|\d+|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.:/-]*|\d+")
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
    r"^\s*(?:async\s+def|def|class|interface|enum|struct|trait|impl|"
    r"function|func|fn|import|from|package|use|#include|module|namespace)\b"
)
TREE_RE = re.compile(r"^(?:[│├└─ ]{2,}|[.A-Za-z0-9_-]+/)\S*")
TEST_RE = re.compile(
    r"(?:=+\s+(?:FAILURES|ERRORS|short test summary)|\b(?:FAILED|ERROR)\b|"
    r"\d+\s+failed\b|AssertionError)",
    re.IGNORECASE,
)
STOPWORDS = {
    "and",
    "are",
    "bash",
    "cat",
    "cd",
    "code",
    "file",
    "for",
    "from",
    "have",
    "into",
    "not",
    "output",
    "sed",
    "that",
    "the",
    "this",
    "with",
    "your",
}
TOKEN_ESTIMATOR = "max-lexical-ascii4-unicode1-v2"


class OutputKind(str, Enum):
    SOURCE = "source"
    DIFF = "diff"
    TRACEBACK = "traceback"
    TEST_LOG = "test_log"
    SEARCH = "search"
    TREE = "tree"
    GENERIC = "generic"


@dataclass(frozen=True)
class EvidenceBlock:
    index: int
    start_line: int
    end_line: int
    text: str
    reasons: frozenset[str]
    terms: frozenset[str]

    @property
    def line_numbers(self) -> tuple[int, ...]:
        return tuple(range(self.start_line, self.end_line + 1))


def estimate_tokens(text: str) -> int:
    """Deterministic token proxy suitable for a no-model cost gate.

    The lexical count is useful for punctuation-heavy logs, but treats a long
    identifier as one token. The character proxy corrects that systematic
    code undercount while charging non-ASCII characters more conservatively.
    This remains local CPU work: no tokenizer, model, or HTTP call occurs.
    """

    lexical_count = len(TOKEN_RE.findall(text))
    ascii_count = sum(character.isascii() for character in text)
    non_ascii_count = len(text) - ascii_count
    character_count = ceil(ascii_count / 4) + non_ascii_count
    return max(lexical_count, character_count)


def _terms(text: str) -> frozenset[str]:
    return frozenset(
        token.lower()
        for token in WORD_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS and not token.isdigit()
    )


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


def classify_output(text: str, *, command: str, path: str) -> OutputKind:
    lines = text.splitlines()
    command_lower = command.lower()
    verb = _shell_verb(command)
    if "git diff" in command_lower or sum(bool(DIFF_RE.search(line)) for line in lines) >= 3:
        return OutputKind.DIFF
    if (
        "traceback (most recent call last)" in text.lower()
        or sum(bool(TRACE_FRAME_RE.search(line)) for line in lines) >= 2
    ):
        return OutputKind.TRACEBACK
    if TEST_RE.search(text) or (
        verb in {"pytest", "tox", "jest", "npm", "pnpm", "yarn", "go", "cargo"}
        and "test" in command_lower
    ):
        return OutputKind.TEST_LOG
    if verb in {"grep", "rg", "ag", "ack", "find"}:
        return OutputKind.SEARCH
    if verb in {"tree", "ls"} and sum(bool(TREE_RE.search(line)) for line in lines[:100]) >= 3:
        return OutputKind.TREE
    suffixes = (
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
    if path.lower().endswith(suffixes):
        return OutputKind.SOURCE
    if len(lines) >= 20 and sum(bool(STRUCTURE_RE.search(line)) for line in lines) >= max(
        2, len(lines) // 50
    ):
        return OutputKind.SOURCE
    return OutputKind.GENERIC


def _line_reasons(line: str, kind: OutputKind) -> frozenset[str]:
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
    allowed = {
        OutputKind.SOURCE: {"structure", "error", "source-location", "diff"},
        OutputKind.DIFF: {"diff", "error", "source-location"},
        OutputKind.TRACEBACK: {"error", "trace-frame", "source-location"},
        OutputKind.TEST_LOG: {"error", "trace-frame", "source-location", "test-failure"},
        OutputKind.SEARCH: {"error", "source-location"},
        OutputKind.TREE: {"error"},
        OutputKind.GENERIC: {"error", "source-location", "test-failure"},
    }[kind]
    return frozenset(values.intersection(allowed))


def _chunks(start: int, end: int, size: int) -> Iterable[tuple[int, int]]:
    while start <= end:
        chunk_end = min(end, start + size - 1)
        yield start, chunk_end
        start = chunk_end + 1


def build_blocks(text: str, *, kind: OutputKind, max_lines: int) -> list[EvidenceBlock]:
    lines = text.splitlines()
    special = {
        line_no: _line_reasons(line, kind)
        for line_no, line in enumerate(lines, start=1)
        if _line_reasons(line, kind)
    }
    ranges: list[tuple[int, int, frozenset[str]]] = []
    cursor = 1
    for line_no, reasons in sorted(special.items()):
        if cursor < line_no:
            ranges.extend(
                (start, end, frozenset()) for start, end in _chunks(cursor, line_no - 1, max_lines)
            )
        ranges.append((line_no, line_no, reasons))
        cursor = line_no + 1
    if cursor <= len(lines):
        ranges.extend(
            (start, end, frozenset()) for start, end in _chunks(cursor, len(lines), max_lines)
        )
    return [
        EvidenceBlock(
            index=index,
            start_line=start,
            end_line=end,
            text="\n".join(lines[start - 1 : end]),
            reasons=reasons,
            terms=_terms("\n".join(lines[start - 1 : end])),
        )
        for index, (start, end, reasons) in enumerate(ranges)
    ]


def _hard_indices(blocks: Sequence[EvidenceBlock], kind: OutputKind) -> set[int]:
    required = {
        OutputKind.SOURCE: {"structure", "error", "source-location"},
        OutputKind.DIFF: {"diff", "error", "source-location"},
        OutputKind.TRACEBACK: {"error", "trace-frame", "source-location"},
        OutputKind.TEST_LOG: {"error", "trace-frame", "source-location", "test-failure"},
        OutputKind.SEARCH: {"error", "source-location"},
        OutputKind.TREE: {"error"},
        OutputKind.GENERIC: {"error", "source-location", "test-failure"},
    }[kind]
    selected = {block.index for block in blocks if required.intersection(block.reasons)}
    if blocks:
        selected.update({0, len(blocks) - 1})
    return selected


def _expand(selected: Iterable[int], *, block_count: int, radius: int) -> set[int]:
    result: set[int] = set()
    for index in selected:
        result.update(range(max(0, index - radius), min(block_count, index + radius + 1)))
    return result


def _render(lines: Sequence[str], kept: set[int], *, signal: PosteriorSignal) -> str:
    command = " ".join(signal.command.split())[:240]
    header = (
        f'<posterior_history_compaction kept_lines="{len(kept)}" original_lines="{len(lines)}">\n'
        "This older observation was shown in full before the next action was chosen. "
        "The retained evidence is selected from that normal follow-up action"
        f"{': ' + command if command else ''}.\n"
        "</posterior_history_compaction>"
    )
    output = [header]
    line_no = 1
    while line_no <= len(lines):
        if line_no in kept:
            output.append(lines[line_no - 1])
            line_no += 1
            continue
        start = line_no
        while line_no <= len(lines) and line_no not in kept:
            line_no += 1
        end = line_no - 1
        label = f"line {start}" if start == end else f"lines {start}-{end}"
        output.append(f"... [posterior-history omitted {label}] ...")
    return "\n".join(output)


def _unchanged(
    text: str,
    *,
    reason: str,
    method: str,
    kind: OutputKind,
    block_count: int = 0,
    hard_block_count: int = 0,
    matched_block_count: int = 0,
    selected_block_count: int = 0,
) -> CompactionResult:
    tokens = estimate_tokens(text)
    lines = text.splitlines()
    return CompactionResult(
        text=text,
        status="skipped",
        reason=reason,
        method=method,
        output_kind=kind.value,
        origin_token_cnt=tokens,
        left_token_cnt=tokens,
        original_line_count=len(lines),
        kept_line_count=len(lines),
        retained_line_numbers=tuple(range(1, len(lines) + 1)),
        block_count=block_count,
        hard_block_count=hard_block_count,
        matched_block_count=matched_block_count,
        selected_block_count=selected_block_count,
    )


def compact_after_followup(
    text: str,
    *,
    causing_command: str,
    causing_path: str,
    posterior: PosteriorSignal,
    config: PosteriorHistoryConfig,
) -> CompactionResult:
    """Return a compact *history view* after the model has seen ``text`` once.

    The original text remains in canonical agent history.  This function never
    sends a request to an LLM, tokenizer, HTTP service, or raw-store endpoint.
    """

    kind = classify_output(text, command=causing_command, path=causing_path)
    method = f"posterior_{config.method}"
    original_tokens = estimate_tokens(text)
    lines = text.splitlines()
    if not text.strip():
        return _unchanged(text, reason="empty-observation", method=method, kind=kind)
    if original_tokens < config.min_input_tokens:
        return _unchanged(text, reason="below-min-input-tokens", method=method, kind=kind)
    if kind == OutputKind.DIFF:
        return _unchanged(text, reason="diff-is-never-compacted", method=method, kind=kind)
    if not posterior.text.strip():
        return _unchanged(text, reason="missing-posterior-signal", method=method, kind=kind)

    blocks = build_blocks(text, kind=kind, max_lines=config.block_max_lines)
    if not blocks:
        return _unchanged(text, reason="no-blocks", method=method, kind=kind)
    hard = _hard_indices(blocks, kind)
    selected = set(hard)
    term_frequency: Counter[str] = Counter(term for block in blocks for term in block.terms)
    # Generic language and common code names ("config", "value", "result")
    # must not make every block look relevant. All posterior evidence,
    # including identifier-shaped words, therefore passes the same local
    # frequency gate. Rare exact symbols such as resolve_model still match.
    max_term_frequency = max(2, (len(blocks) + 9) // 10)
    posterior_terms = {
        term
        for term in _terms(posterior.text)
        if 0 < term_frequency.get(term, 0) <= max_term_frequency
    }
    matches = {block.index for block in blocks if posterior_terms.intersection(block.terms)}
    used_matches = matches if config.method == "adaptive" else set()
    if config.method == "adaptive":
        # Without an actual later-action match, preserving only a generic
        # skeleton is too risky.  Keep the complete historical observation.
        if not matches:
            return _unchanged(
                text,
                reason="no-posterior-match",
                method=method,
                kind=kind,
                block_count=len(blocks),
                hard_block_count=len(hard),
                selected_block_count=len(hard),
            )
        selected.update(matches)
        radius = (
            1
            if kind == OutputKind.SOURCE
            else 2
            if kind
            in {
                OutputKind.TRACEBACK,
                OutputKind.TEST_LOG,
            }
            else 0
        )
        selected.update(_expand(matches, block_count=len(blocks), radius=radius))
    elif kind in {OutputKind.TRACEBACK, OutputKind.TEST_LOG}:
        selected.update(_expand(hard, block_count=len(blocks), radius=1))

    kept = {
        line_no for block in blocks if block.index in selected for line_no in block.line_numbers
    }
    if not kept or len(kept) >= len(lines):
        if len(hard) == len(blocks):
            reason = "no-safe-reduction-hard-skeleton"
        elif config.method == "adaptive":
            reason = "no-safe-reduction-posterior-expanded"
        else:
            reason = "no-safe-reduction-structural-neighborhood"
        return _unchanged(
            text,
            reason=reason,
            method=method,
            kind=kind,
            block_count=len(blocks),
            hard_block_count=len(hard),
            matched_block_count=len(used_matches),
            selected_block_count=len(selected),
        )
    line_retention = len(kept) / len(lines)
    if line_retention > config.max_retention_ratio:
        return _unchanged(
            text,
            reason="retention-above-cost-gate",
            method=method,
            kind=kind,
            block_count=len(blocks),
            hard_block_count=len(hard),
            matched_block_count=len(used_matches),
            selected_block_count=len(selected),
        )
    compacted = _render(lines, kept, signal=posterior)
    if len(compacted) > config.max_output_chars:
        return _unchanged(
            text,
            reason="output-cap-not-safe",
            method=method,
            kind=kind,
            block_count=len(blocks),
            hard_block_count=len(hard),
            matched_block_count=len(used_matches),
            selected_block_count=len(selected),
        )
    left_tokens = estimate_tokens(compacted)
    if original_tokens - left_tokens < config.min_savings_tokens:
        return _unchanged(
            text,
            reason="insufficient-token-savings",
            method=method,
            kind=kind,
            block_count=len(blocks),
            hard_block_count=len(hard),
            matched_block_count=len(used_matches),
            selected_block_count=len(selected),
        )
    return CompactionResult(
        text=compacted,
        status="compacted",
        reason="posterior-action-guided",
        method=method,
        output_kind=kind.value,
        origin_token_cnt=original_tokens,
        left_token_cnt=left_tokens,
        original_line_count=len(lines),
        kept_line_count=len(kept),
        retained_line_numbers=tuple(sorted(kept)),
        block_count=len(blocks),
        hard_block_count=len(hard),
        matched_block_count=len(used_matches),
        selected_block_count=len(selected),
    )
