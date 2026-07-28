from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\w\s]", re.UNICODE)
IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.:-]*")
STRUCTURE_RE = re.compile(
    r"^\s*(?:async\s+def|def|class|interface|enum|struct|trait|impl|"
    r"function|func|fn|export\s+(?:default\s+)?(?:class|function)|"
    r"import|from|package|use|#include)\b"
)
ERROR_RE = re.compile(
    r"(?:traceback|error|exception|failed|failure|assert(?:ion)?|"
    r"panic|segmentation fault|caused by|fatal|warning)\b",
    re.IGNORECASE,
)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def identifiers(text: str) -> set[str]:
    return {token.lower() for token in IDENTIFIER_RE.findall(text) if len(token) > 1}


def build_query(
    query: str,
    *,
    path: str | None = None,
    recent_context: Iterable[str] = (),
) -> str:
    parts = [query.strip()]
    parts.extend(item.strip() for item in recent_context if item.strip())
    if path:
        path_obj = Path(path)
        parts.extend((path_obj.name, path_obj.stem))
    return " ".join(part for part in parts if part)


def structural_anchor_lines(lines: Sequence[str]) -> set[int]:
    return {line_no for line_no, line in enumerate(lines, start=1) if STRUCTURE_RE.search(line)}


def error_anchor_lines(lines: Sequence[str]) -> set[int]:
    return {line_no for line_no, line in enumerate(lines, start=1) if ERROR_RE.search(line)}


@dataclass(frozen=True)
class TextBlock:
    start_line: int
    end_line: int
    text: str
    kind: str = "text"

    @property
    def line_numbers(self) -> tuple[int, ...]:
        return tuple(range(self.start_line, self.end_line + 1))


def python_ast_blocks(text: str) -> list[TextBlock]:
    lines = text.splitlines()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    blocks: list[TextBlock] = []
    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            continue
        end_line = int(getattr(node, "end_lineno", node.lineno))
        blocks.append(
            TextBlock(
                start_line=int(node.lineno),
                end_line=end_line,
                text="\n".join(lines[node.lineno - 1 : end_line]),
                kind=type(node).__name__,
            )
        )
    return sorted(blocks, key=lambda block: (block.start_line, block.end_line))


def paragraph_blocks(text: str, *, max_lines: int = 24) -> list[TextBlock]:
    lines = text.splitlines()
    if not lines:
        return []
    blocks: list[TextBlock] = []
    start = 1
    for line_no, line in enumerate(lines, start=1):
        boundary = not line.strip() or line_no - start + 1 >= max_lines
        if boundary:
            end = line_no if line.strip() else line_no - 1
            if end >= start:
                blocks.append(
                    TextBlock(
                        start_line=start,
                        end_line=end,
                        text="\n".join(lines[start - 1 : end]),
                    )
                )
            start = line_no + 1
    if start <= len(lines):
        blocks.append(
            TextBlock(
                start_line=start,
                end_line=len(lines),
                text="\n".join(lines[start - 1 :]),
            )
        )
    return blocks


def code_aware_blocks(text: str) -> list[TextBlock]:
    ast_blocks = python_ast_blocks(text)
    if ast_blocks:
        return ast_blocks
    return paragraph_blocks(text)
