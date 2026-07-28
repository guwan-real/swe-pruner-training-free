from __future__ import annotations

import ast
import re
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from tf_pruning.protocol import (
    PruningRequest,
    PruningResult,
    coerce_line_scores,
)
from tf_pruning.selection import (
    expand_line_numbers,
    render_pruned_text,
    select_line_numbers,
)
from tf_pruning.text import (
    error_anchor_lines,
    identifiers,
    structural_anchor_lines,
)

_TRACEBACK_FRAME_RE = re.compile(
    r'^\s*File\s+"[^"]+",\s+line\s+\d+|'
    r"^\s*at\s+.+\([^():]+:\d+(?::\d+)?\)",
)
_TRACEBACK_EXCEPTION_RE = re.compile(
    r"^\s*(?:[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception)|"
    r"panic|fatal|segmentation fault)\s*:",
    re.IGNORECASE,
)
_FILE_LINE_RE = re.compile(
    r"(?:^|\s)(?:[A-Za-z]:)?[^:\s]+\.[A-Za-z0-9_]+:\d+(?::\d+)?",
)
_GREP_LINE_RE = re.compile(
    r"^(?:[A-Za-z]:)?[^:\n]+:\d+(?::|-)|^\s*\d+(?::|-)",
)
_CODE_DECLARATION_RE = re.compile(
    r"(?:^|:\s*)\s*(?:async\s+def|def|class|interface|enum|struct|"
    r"trait|impl|function|func|fn)\s+[A-Za-z_]",
)
_DIFF_HUNK_RE = re.compile(r"^@@(?:@)?\s")
_TEST_SIGNAL_RE = re.compile(
    r"(?:^|\b)(?:FAILED|FAILURE|FAILURES|ERRORS?|"
    r"AssertionError|assertion failed|short test summary info|"
    r"\d+\s+failed)(?:\b|:)",
    re.IGNORECASE,
)
_TREE_BRANCH_RE = re.compile(r"^(?P<prefix>[\s│|]*)(?:├──|└──|\+--|`--)\s*")
_QUERY_PART_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


_TOOL_ALIASES = {
    "cat": "source",
    "file": "source",
    "read": "source",
    "read_file": "source",
    "sed": "source",
    "source_file": "source",
    "rg": "grep",
    "ripgrep": "grep",
    "search": "grep",
    "git_diff": "diff",
    "pytest": "test_log",
    "tests": "test_log",
    "test": "test_log",
    "ls": "tree",
    "find": "tree",
    "directory": "tree",
}
_SUPPORTED_TOOL_TYPES = {
    "source",
    "grep",
    "traceback",
    "diff",
    "test_log",
    "tree",
    "generic",
}


@dataclass(frozen=True)
class ExecutionASTConfig:
    """Fixed rule strengths for the execution-signal pruner."""

    query_match_score: float = 9.0
    critical_signal_score: float = 10.0
    ast_hit_body_score: float = 8.0
    ast_neighbor_body_score: float = 5.0
    skeleton_score: float = 3.0
    context_score: float = 1.0
    fallback_score: float = 0.01
    preserve_python_skeleton: bool = True
    expand_hit_bodies: bool = True
    expand_one_hop: bool = True
    tree_shallow_depth: int = 1
    show_line_numbers: bool = True

    def __post_init__(self) -> None:
        if self.tree_shallow_depth < 0:
            raise ValueError("tree_shallow_depth must be non-negative")
        for item in fields(self):
            if item.name.endswith("_score") and getattr(self, item.name) < 0:
                raise ValueError(f"{item.name} must be non-negative")

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, Any] | None,
    ) -> "ExecutionASTConfig":
        if config is None:
            return cls()
        values = dict(config)
        scores = values.pop("scores", None)
        if scores is not None:
            if not isinstance(scores, Mapping):
                raise TypeError("scores must be a mapping")
            for name, value in scores.items():
                key = str(name)
                if not key.endswith("_score"):
                    key = f"{key}_score"
                values.setdefault(key, value)
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"unknown execution AST config keys: {unknown}")
        return cls(**values)


def _metadata_hint(request: PruningRequest) -> str:
    keys = ("tool", "tool_name", "command", "cmd", "operation")
    return " ".join(
        str(request.metadata.get(key, "")) for key in keys if request.metadata.get(key)
    ).lower()


def _normalise_tool_type(tool_type: str) -> str:
    normalised = tool_type.strip().lower().replace("-", "_").replace(" ", "_")
    normalised = _TOOL_ALIASES.get(normalised, normalised)
    return normalised if normalised in _SUPPORTED_TOOL_TYPES else "generic"


def detect_tool_type(request: PruningRequest) -> str:
    """Infer a supported observation type without executing external tools."""

    if request.tool_type.strip().lower() != "auto":
        return _normalise_tool_type(request.tool_type)

    hint = _metadata_hint(request)
    if re.search(r"(?:^|\s)(?:rg|grep|ripgrep|ag)(?:\s|$)", hint):
        return "grep"
    if re.search(r"\bgit\s+diff\b|\bdiff\b", hint):
        return "diff"
    if re.search(r"\b(?:pytest|unittest|jest|vitest|cargo\s+test|go\s+test)\b", hint):
        return "test_log"
    if re.search(r"(?:^|\s)(?:tree|find|ls)(?:\s|$)", hint):
        return "tree"
    if re.search(r"(?:^|\s)(?:cat|sed|head|tail|read_file)(?:\s|$)", hint):
        return "source"

    text = request.text
    lines = text.splitlines()
    if (
        "Traceback (most recent call last):" in text
        or sum(bool(_TRACEBACK_FRAME_RE.search(line)) for line in lines) >= 2
    ):
        return "traceback"
    if "diff --git " in text or (
        any(_DIFF_HUNK_RE.match(line) for line in lines)
        and any(line.startswith(("--- ", "+++ ")) for line in lines)
    ):
        return "diff"
    if _TEST_SIGNAL_RE.search(text):
        return "test_log"
    if any(_GREP_LINE_RE.search(line) for line in lines):
        return "grep"
    if any(_TREE_BRANCH_RE.match(line) for line in lines):
        return "tree"

    suffix = Path(request.path).suffix.lower() if request.path else ""
    if suffix in {
        ".py",
        ".pyi",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".java",
        ".go",
        ".rs",
        ".c",
        ".cc",
        ".cpp",
        ".h",
        ".hpp",
    }:
        return "source"
    if structural_anchor_lines(lines):
        return "source"
    return "generic"


def _query_symbols(request: PruningRequest) -> set[str]:
    values = [request.query, *request.recent_context]
    symbols: set[str] = set()
    for value in values:
        for item in identifiers(value):
            symbols.add(item.lower())
            symbols.update(part.lower() for part in re.split(r"[.:\-/]+", item) if len(part) > 1)
        symbols.update(
            match.group(0).lower()
            for match in _QUERY_PART_RE.finditer(value)
            if len(match.group(0)) > 1
        )
    return symbols


def _called_names(node: ast.AST) -> set[str]:
    called: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        function = child.func
        if isinstance(function, ast.Name):
            called.add(function.id.lower())
        elif isinstance(function, ast.Attribute):
            called.add(function.attr.lower())
    return called


def _node_start(node: ast.AST) -> int:
    decorator_lines = [
        int(decorator.lineno)
        for decorator in getattr(node, "decorator_list", ())
        if hasattr(decorator, "lineno")
    ]
    return min([int(getattr(node, "lineno", 1)), *decorator_lines])


def _node_range(node: ast.AST, line_count: int) -> range:
    start = max(1, _node_start(node))
    end = min(line_count, int(getattr(node, "end_lineno", start)))
    return range(start, end + 1)


def _tree_depth(line: str) -> int:
    branch = _TREE_BRANCH_RE.match(line)
    if branch:
        prefix = branch.group("prefix").replace("│", " ")
        return max(1, len(prefix.expandtabs(4)) // 4 + 1)
    stripped = line.lstrip()
    indentation = len(line) - len(stripped)
    return indentation // 2


class _Signals:
    def __init__(self, line_count: int, fallback_score: float) -> None:
        self.scores = [fallback_score] * line_count
        self.reasons: dict[int, list[str]] = defaultdict(list)
        self.mandatory: set[int] = set()
        self.expansion_seeds: set[int] = set()

    def add(
        self,
        line_numbers: int | Iterable[int],
        score: float,
        reason: str,
        *,
        mandatory: bool = False,
        expand: bool = False,
    ) -> None:
        numbers = (line_numbers,) if isinstance(line_numbers, int) else line_numbers
        for line_no in numbers:
            if not 1 <= line_no <= len(self.scores):
                continue
            self.scores[line_no - 1] += score
            if reason not in self.reasons[line_no]:
                self.reasons[line_no].append(reason)
            if mandatory:
                self.mandatory.add(line_no)
            if expand:
                self.expansion_seeds.add(line_no)


class ExecutionASTPruner:
    """Preserve execution evidence and expand Python symbols structurally."""

    name = "execution_ast"

    def __init__(self, config: ExecutionASTConfig | None = None) -> None:
        self.config = config or ExecutionASTConfig()

    def prune(self, request: PruningRequest) -> PruningResult:
        started_at = time.perf_counter()
        lines = request.lines
        if not lines:
            return PruningResult(
                method=self.name,
                original_line_count=0,
                kept_line_numbers=(),
                pruned_text="",
                latency_ms=(time.perf_counter() - started_at) * 1000.0,
                metadata={
                    "config": asdict(self.config),
                    "detected_tool_type": detect_tool_type(request),
                },
                request_id=request.request_id,
            )

        detected_type = detect_tool_type(request)
        symbols = _query_symbols(request)
        signals = _Signals(len(lines), self.config.fallback_score)

        self._add_query_matches(lines, symbols, signals)
        ast_parsed = False
        ast_expanded_symbols: set[str] = set()
        if detected_type == "source":
            ast_parsed, ast_expanded_symbols = self._source_rules(
                request.text,
                lines,
                symbols,
                signals,
            )
        elif detected_type == "grep":
            self._grep_rules(lines, signals)
        elif detected_type == "traceback":
            self._traceback_rules(lines, signals)
        elif detected_type == "diff":
            self._diff_rules(lines, signals)
        elif detected_type == "test_log":
            self._test_log_rules(lines, signals)
        elif detected_type == "tree":
            self._tree_rules(lines, symbols, signals)
        else:
            self._generic_rules(lines, signals)

        context_lines = expand_line_numbers(
            signals.expansion_seeds,
            line_count=len(lines),
            window=request.budget.context_window,
        )
        for line_no in context_lines - signals.expansion_seeds:
            signals.add(
                line_no,
                self.config.context_score,
                "execution_context",
            )

        kept = select_line_numbers(
            signals.scores,
            request.budget,
            mandatory=signals.mandatory,
            expansion_seeds=signals.expansion_seeds,
        )
        return PruningResult(
            method=self.name,
            original_line_count=len(lines),
            kept_line_numbers=kept,
            pruned_text=render_pruned_text(
                lines,
                kept,
                show_line_numbers=self.config.show_line_numbers,
            ),
            line_scores=coerce_line_scores(signals.scores, signals.reasons),
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
            metadata={
                "config": asdict(self.config),
                "detected_tool_type": detected_type,
                "python_ast_parsed": ast_parsed,
                "query_symbols": sorted(symbols),
                "ast_expanded_symbols": sorted(ast_expanded_symbols),
                "mandatory_lines": sorted(signals.mandatory),
                "expansion_seed_lines": sorted(signals.expansion_seeds),
                "training_free": True,
            },
            request_id=request.request_id,
        )

    def _add_query_matches(
        self,
        lines: Sequence[str],
        symbols: set[str],
        signals: _Signals,
    ) -> None:
        for line_no, line in enumerate(lines, start=1):
            line_symbols = identifiers(line)
            leaf_symbols = {
                part for symbol in line_symbols for part in re.split(r"[.:\-/]+", symbol)
            }
            if symbols & (line_symbols | leaf_symbols):
                signals.add(
                    line_no,
                    self.config.query_match_score,
                    "query_symbol_match",
                    mandatory=True,
                    expand=True,
                )

    def _source_rules(
        self,
        text: str,
        lines: Sequence[str],
        symbols: set[str],
        signals: _Signals,
    ) -> tuple[bool, set[str]]:
        for line_no in error_anchor_lines(lines):
            signals.add(
                line_no,
                self.config.critical_signal_score,
                "source_error",
                mandatory=True,
                expand=True,
            )
        try:
            tree = ast.parse(text)
        except SyntaxError:
            for line_no in structural_anchor_lines(lines):
                signals.add(
                    line_no,
                    self.config.skeleton_score,
                    "structural_skeleton",
                    mandatory=True,
                    expand=True,
                )
            return False, set()

        definitions = [
            node
            for node in ast.walk(tree)
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            )
        ]
        imports = [
            node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        if self.config.preserve_python_skeleton:
            for node in imports:
                signals.add(
                    _node_range(node, len(lines)),
                    self.config.skeleton_score,
                    "ast_import",
                    mandatory=True,
                )
            for node in definitions:
                signature_line = int(getattr(node, "lineno", _node_start(node)))
                signals.add(
                    signature_line,
                    self.config.skeleton_score,
                    "ast_signature",
                    mandatory=True,
                    expand=True,
                )
                if _node_start(node) < signature_line:
                    signals.add(
                        range(_node_start(node), signature_line),
                        self.config.skeleton_score,
                        "ast_decorator",
                        mandatory=True,
                    )

        definition_map: dict[str, list[ast.AST]] = defaultdict(list)
        calls_by_node: dict[int, set[str]] = {}
        parent_by_node: dict[ast.AST, ast.AST] = {
            child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)
        }
        referenced_by_definition: dict[int, set[str]] = defaultdict(set)
        definition_ids = {id(node): node for node in definitions}
        for child in ast.walk(tree):
            referenced_name: str | None = None
            if isinstance(child, ast.Name):
                referenced_name = child.id.lower()
            elif isinstance(child, ast.Attribute):
                referenced_name = child.attr.lower()
            elif isinstance(child, ast.arg):
                referenced_name = child.arg.lower()
            if referenced_name is None:
                continue
            ancestor = parent_by_node.get(child)
            while ancestor is not None and id(ancestor) not in definition_ids:
                ancestor = parent_by_node.get(ancestor)
            if ancestor is not None:
                referenced_by_definition[id(ancestor)].add(referenced_name)
        for node in definitions:
            name = str(getattr(node, "name", "")).lower()
            if name:
                definition_map[name].append(node)
            calls_by_node[id(node)] = _called_names(node)

        direct_nodes = [
            node
            for node in definitions
            if str(getattr(node, "name", "")).lower() in symbols
            or bool(referenced_by_definition[id(node)] & symbols)
        ]
        direct_names = {
            str(getattr(node, "name", "")).lower()
            for node in direct_nodes
            if getattr(node, "name", "")
        }
        expanded_names = set(direct_names)
        if self.config.expand_hit_bodies:
            for node in direct_nodes:
                signals.add(
                    _node_range(node, len(lines)),
                    self.config.ast_hit_body_score,
                    "ast_hit_body",
                    mandatory=True,
                    expand=True,
                )

        if self.config.expand_one_hop and direct_nodes:
            neighbour_names: set[str] = set()
            for node in direct_nodes:
                neighbour_names.update(calls_by_node[id(node)] & set(definition_map))
            for node in definitions:
                name = str(getattr(node, "name", "")).lower()
                if calls_by_node[id(node)] & direct_names:
                    neighbour_names.add(name)
            neighbour_names.difference_update(direct_names)
            expanded_names.update(neighbour_names)
            for name in neighbour_names:
                for node in definition_map[name]:
                    signals.add(
                        _node_range(node, len(lines)),
                        self.config.ast_neighbor_body_score,
                        "ast_one_hop_body",
                        mandatory=True,
                        expand=True,
                    )
        return True, expanded_names

    def _grep_rules(self, lines: Sequence[str], signals: _Signals) -> None:
        structures = structural_anchor_lines(lines)
        for line_no, line in enumerate(lines, start=1):
            if _GREP_LINE_RE.search(line):
                signals.add(
                    line_no,
                    self.config.critical_signal_score,
                    "grep_hit",
                    mandatory=True,
                    expand=True,
                )
            if line_no in structures or _CODE_DECLARATION_RE.search(line):
                signals.add(
                    line_no,
                    self.config.skeleton_score,
                    "grep_definition",
                    mandatory=True,
                )

    def _traceback_rules(
        self,
        lines: Sequence[str],
        signals: _Signals,
    ) -> None:
        errors = error_anchor_lines(lines)
        for line_no, line in enumerate(lines, start=1):
            if "Traceback (most recent call last):" in line:
                signals.add(
                    line_no,
                    self.config.critical_signal_score,
                    "traceback_header",
                    mandatory=True,
                )
            if _TRACEBACK_FRAME_RE.search(line):
                signals.add(
                    line_no,
                    self.config.critical_signal_score,
                    "traceback_frame",
                    mandatory=True,
                    expand=True,
                )
            if _TRACEBACK_EXCEPTION_RE.search(line):
                signals.add(
                    line_no,
                    self.config.critical_signal_score + self.config.skeleton_score,
                    "traceback_exception",
                    mandatory=True,
                    expand=True,
                )
            if line_no in errors:
                signals.add(
                    line_no,
                    self.config.critical_signal_score,
                    "traceback_error",
                    mandatory=True,
                    expand=True,
                )

    def _diff_rules(self, lines: Sequence[str], signals: _Signals) -> None:
        for line_no, line in enumerate(lines, start=1):
            if line.startswith(("diff --git ", "index ", "--- ", "+++ ")) or _DIFF_HUNK_RE.match(
                line
            ):
                signals.add(
                    line_no,
                    self.config.skeleton_score,
                    "diff_header",
                    mandatory=True,
                    expand=bool(_DIFF_HUNK_RE.match(line)),
                )
            elif line.startswith(("+", "-")):
                signals.add(
                    line_no,
                    self.config.critical_signal_score,
                    "diff_change",
                    mandatory=True,
                    expand=True,
                )

    def _test_log_rules(
        self,
        lines: Sequence[str],
        signals: _Signals,
    ) -> None:
        errors = error_anchor_lines(lines)
        for line_no, line in enumerate(lines, start=1):
            if _TEST_SIGNAL_RE.search(line):
                signals.add(
                    line_no,
                    self.config.critical_signal_score,
                    "test_failure",
                    mandatory=True,
                    expand=True,
                )
            if _FILE_LINE_RE.search(line):
                signals.add(
                    line_no,
                    self.config.critical_signal_score,
                    "test_file_line",
                    mandatory=True,
                    expand=True,
                )
            if line_no in errors:
                signals.add(
                    line_no,
                    self.config.critical_signal_score,
                    "test_error",
                    mandatory=True,
                    expand=True,
                )

    def _tree_rules(
        self,
        lines: Sequence[str],
        symbols: set[str],
        signals: _Signals,
    ) -> None:
        depths = [_tree_depth(line) for line in lines]
        hit_lines: list[int] = []
        for line_no, (line, depth) in enumerate(
            zip(lines, depths),
            start=1,
        ):
            if depth <= self.config.tree_shallow_depth:
                signals.add(
                    line_no,
                    self.config.skeleton_score,
                    "tree_shallow",
                    mandatory=True,
                )
            lowered = line.lower()
            if any(symbol in lowered for symbol in symbols):
                hit_lines.append(line_no)
                signals.add(
                    line_no,
                    self.config.critical_signal_score,
                    "tree_query_hit",
                    mandatory=True,
                    expand=True,
                )

        for hit_line in hit_lines:
            current_depth = depths[hit_line - 1]
            for candidate in range(hit_line - 1, 0, -1):
                candidate_depth = depths[candidate - 1]
                if candidate_depth < current_depth:
                    signals.add(
                        candidate,
                        self.config.skeleton_score,
                        "tree_ancestor",
                        mandatory=True,
                    )
                    current_depth = candidate_depth
                    if current_depth <= 0:
                        break

    def _generic_rules(
        self,
        lines: Sequence[str],
        signals: _Signals,
    ) -> None:
        for line_no in structural_anchor_lines(lines):
            signals.add(
                line_no,
                self.config.skeleton_score,
                "generic_structure",
                mandatory=True,
                expand=True,
            )
        for line_no in error_anchor_lines(lines):
            signals.add(
                line_no,
                self.config.critical_signal_score,
                "generic_error",
                mandatory=True,
                expand=True,
            )
