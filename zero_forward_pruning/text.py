from __future__ import annotations

import re
import shlex
from enum import Enum
from pathlib import PurePosixPath

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]*|\d+|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.:/-]*|\d+")
IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")
PATH_RE = re.compile(
    r"(?:^|[\s\"'`(])((?:\.{0,2}/)?[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+"
    r"|[A-Za-z0-9_.-]+\.(?:py|js|jsx|ts|tsx|go|rs|java|c|cc|cpp|h|hpp|rb|php|sh|yaml|yml|toml|json))"
)
LOCATION_RE = re.compile(
    r"(?P<path>[A-Za-z0-9_./-]+\.(?:py|js|jsx|ts|tsx|go|rs|java|c|cc|cpp|h|hpp|rb|php|sh))"
    r"(?::(?P<line>\d+))?"
)
ERROR_RE = re.compile(
    r"\b(?:traceback|error|exception|failed|failure|fatal|panic|assertion|segfault|warning)\b",
    re.IGNORECASE,
)
TRACE_FRAME_RE = re.compile(r"^\s*(?:File\s+\"[^\"]+\",\s+line\s+\d+|at\s+\S+.*:\d+)")
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


class OutputKind(str, Enum):
    SOURCE = "source"
    DIFF = "diff"
    TRACEBACK = "traceback"
    TEST_LOG = "test_log"
    SEARCH = "search"
    TREE = "tree"
    GENERIC = "generic"


def estimate_tokens(text: str) -> int:
    """Cheap deterministic token proxy; it never invokes the served tokenizer."""

    return len(TOKEN_RE.findall(text))


def terms(text: str) -> tuple[str, ...]:
    return tuple(
        token.lower()
        for token in WORD_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS and not token.isdigit()
    )


def identifiers(text: str) -> set[str]:
    return {
        token.lower() for token in IDENTIFIER_RE.findall(text) if token.lower() not in STOPWORDS
    }


def extract_paths(text: str) -> tuple[str, ...]:
    values: list[str] = []
    for match in PATH_RE.finditer(text):
        path = match.group(1).strip(".,:;")
        if path and path not in values:
            values.append(path)
    return tuple(values)


def shell_verb(command: str) -> str:
    if not command.strip():
        return ""
    try:
        words = shlex.split(command)
    except ValueError:
        words = command.split()
    wrappers = {"env", "sudo", "timeout", "command", "xargs"}
    for word in words:
        base = PurePosixPath(word).name.lower()
        if "=" in word and not word.startswith(("/", "./", "../")):
            continue
        if base in wrappers or base.startswith("-"):
            continue
        return base
    return ""


def classify_output(code: str, *, command: str = "", path: str = "") -> OutputKind:
    lines = code.splitlines()
    command_lower = command.lower()
    verb = shell_verb(command)
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
    if "git diff" in command_lower or sum(bool(DIFF_RE.search(line)) for line in lines) >= 3:
        return OutputKind.DIFF
    if (
        "traceback (most recent call last)" in code.lower()
        or sum(bool(TRACE_FRAME_RE.search(line)) for line in lines) >= 2
    ):
        return OutputKind.TRACEBACK
    if verb in {"pytest", "tox", "jest", "npm", "pnpm", "yarn", "go", "cargo"} and (
        TEST_RE.search(code) or "test" in command_lower
    ):
        return OutputKind.TEST_LOG
    if TEST_RE.search(code):
        return OutputKind.TEST_LOG
    if verb in {"grep", "rg", "ag", "ack", "find"}:
        return OutputKind.SEARCH
    if verb in {"tree", "ls"} and sum(bool(TREE_RE.search(line)) for line in lines[:100]) >= 3:
        return OutputKind.TREE
    inferred_paths = (path, *extract_paths(command))
    if any(value.lower().endswith(source_suffixes) for value in inferred_paths if value):
        return OutputKind.SOURCE
    structure_lines = sum(bool(STRUCTURE_RE.search(line)) for line in lines)
    if len(lines) >= 20 and structure_lines >= max(2, len(lines) // 50):
        return OutputKind.SOURCE
    return OutputKind.GENERIC


def line_reasons(line: str) -> tuple[str, ...]:
    reasons: list[str] = []
    if ERROR_RE.search(line):
        reasons.append("error")
    if TRACE_FRAME_RE.search(line):
        reasons.append("trace-frame")
    if DIFF_RE.search(line):
        reasons.append("diff")
    if STRUCTURE_RE.search(line):
        reasons.append("structure")
    if LOCATION_RE.search(line):
        reasons.append("source-location")
    if TEST_RE.search(line):
        reasons.append("test-failure")
    return tuple(reasons)
