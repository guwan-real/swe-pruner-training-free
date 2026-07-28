from __future__ import annotations

import json

import pytest

from tasks.execution_ast import (
    ExecutionASTPruner,
    build_pruner,
    detect_tool_type,
)
from tf_pruning.protocol import BudgetConfig, PruningRequest


@pytest.mark.parametrize(
    ("pruning_request", "expected"),
    [
        (
            PruningRequest(
                text="def run():\n    return 1",
                path="src/app.py",
            ),
            "source",
        ),
        (
            PruningRequest(
                text="src/a.py:3:first\nsrc/b.py:9:second",
            ),
            "grep",
        ),
        (
            PruningRequest(
                text=(
                    "Traceback (most recent call last):\n"
                    '  File "app.py", line 3, in run\n'
                    "ValueError: bad input"
                ),
            ),
            "traceback",
        ),
        (
            PruningRequest(
                text=("diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new"),
            ),
            "diff",
        ),
        (
            PruningRequest(
                text="================ FAILURES ================\nFAILED tests/test_a.py::test_a",
            ),
            "test_log",
        ),
        (
            PruningRequest(
                text="project\n├── src\n│   └── app.py\n└── tests",
            ),
            "tree",
        ),
    ],
)
def test_auto_tool_detection(
    pruning_request: PruningRequest,
    expected: str,
) -> None:
    assert detect_tool_type(pruning_request) == expected


def test_metadata_detection_precedes_ambiguous_text() -> None:
    request = PruningRequest(
        text="src/a.py:1:FAILED is a literal in source",
        metadata={"command": "rg FAILED src"},
    )
    assert detect_tool_type(request) == "grep"


def test_python_ast_keeps_skeleton_hit_body_and_one_hop_neighbour() -> None:
    source = "\n".join(
        [
            "import json",
            "",
            "def helper(value):",
            "    return json.dumps(value)",
            "",
            "def target(value):",
            "    prepared = {'value': value}",
            "    return helper(prepared)",
            "",
            "def unrelated():",
            "    return 'unused'",
            "",
        ]
    )
    request = PruningRequest(
        text=source,
        query="inspect target",
        tool_type="auto",
        path="src/service.py",
        budget=BudgetConfig(
            keep_ratio=0.75,
            no_prune_below=0,
            context_window=1,
        ),
        request_id="ast-1",
    )

    result = build_pruner().prune(request)

    assert result.method == "execution_ast"
    assert result.request_id == "ast-1"
    assert result.metadata["detected_tool_type"] == "source"
    assert result.metadata["python_ast_parsed"] is True
    assert result.metadata["ast_expanded_symbols"] == ["helper", "target"]
    assert {1, 3, 4, 6, 7, 8, 10}.issubset(result.kept_line_numbers)
    assert "ast_hit_body" in result.line_scores[6].reasons
    assert "ast_one_hop_body" in result.line_scores[3].reasons
    assert result.metadata["training_free"] is True


def test_python_ast_expands_definition_containing_hit_variable() -> None:
    source = "\n".join(
        [
            "def first():",
            "    prepared_payload = build_payload()",
            "    return prepared_payload",
            "",
            "def second():",
            "    return None",
        ]
    )
    request = PruningRequest(
        text=source,
        query="prepared_payload",
        tool_type="source",
        budget=BudgetConfig(
            keep_ratio=0.67,
            no_prune_below=0,
            context_window=0,
        ),
    )

    result = ExecutionASTPruner().prune(request)

    assert result.metadata["ast_expanded_symbols"] == ["first"]
    assert {1, 2, 3}.issubset(result.kept_line_numbers)
    assert "ast_hit_body" in result.line_scores[2].reasons


@pytest.mark.parametrize(
    ("tool_type", "text", "reason"),
    [
        (
            "grep",
            "src/a.py:2:target\nsrc/b.py:8:target\nignored",
            "grep_hit",
        ),
        (
            "traceback",
            (
                "Traceback (most recent call last):\n"
                '  File "service.py", line 8, in target\n'
                "    call()\n"
                "RuntimeError: broken"
            ),
            "traceback_frame",
        ),
        (
            "diff",
            (
                "diff --git a/a.py b/a.py\n"
                "--- a/a.py\n+++ b/a.py\n"
                "@@ -1,2 +1,2 @@\n"
                "-old\n+new\n context"
            ),
            "diff_change",
        ),
        (
            "test_log",
            (
                "session starts\n"
                "tests/test_service.py:21: AssertionError\n"
                "FAILED tests/test_service.py::test_target\n"
                "1 failed"
            ),
            "test_failure",
        ),
    ],
)
def test_execution_rules_preserve_critical_lines(
    tool_type: str,
    text: str,
    reason: str,
) -> None:
    request = PruningRequest(
        text=text,
        tool_type=tool_type,
        budget=BudgetConfig(
            keep_ratio=0.75,
            no_prune_below=0,
            context_window=1,
        ),
    )

    result = ExecutionASTPruner().prune(request)

    assert result.metadata["detected_tool_type"] == tool_type
    assert any(
        reason in score.reasons and score.line_no in result.kept_line_numbers
        for score in result.line_scores
    )
    assert result.kept_line_count <= request.budget.target_lines(len(request.lines))


def test_traceback_exception_and_prefixed_grep_definition_are_anchors() -> None:
    traceback_result = ExecutionASTPruner().prune(
        PruningRequest(
            text=(
                "Traceback (most recent call last):\n"
                '  File "service.py", line 8, in target\n'
                "ValueError: broken"
            ),
            tool_type="traceback",
            budget=BudgetConfig(
                keep_ratio=1 / 3,
                no_prune_below=0,
                context_window=0,
            ),
        )
    )
    assert traceback_result.kept_line_numbers == (3,)
    assert "traceback_exception" in traceback_result.line_scores[2].reasons

    grep_result = ExecutionASTPruner().prune(
        PruningRequest(
            text="src/a.py:2:def target():\nplain context",
            tool_type="grep",
            budget=BudgetConfig(
                keep_ratio=0.5,
                no_prune_below=0,
                context_window=0,
            ),
        )
    )
    assert grep_result.kept_line_numbers == (1,)
    assert "grep_definition" in grep_result.line_scores[0].reasons


def test_tree_keeps_query_hit_and_ancestors() -> None:
    tree = "\n".join(
        [
            "project",
            "├── docs",
            "│   └── guide.md",
            "├── src",
            "│   ├── core",
            "│   │   └── target_service.py",
            "│   └── cli.py",
            "└── tests",
        ]
    )
    request = PruningRequest(
        text=tree,
        query="target_service",
        tool_type="tree",
        budget=BudgetConfig(
            keep_ratio=0.75,
            no_prune_below=0,
            context_window=0,
        ),
    )

    result = ExecutionASTPruner().prune(request)

    assert 6 in result.kept_line_numbers
    assert {1, 4, 5}.issubset(result.kept_line_numbers)
    assert "tree_query_hit" in result.line_scores[5].reasons
    assert "tree_ancestor" in result.line_scores[4].reasons


def test_cli_and_nested_score_config(tmp_path) -> None:
    from tasks.execution_ast.cli import main

    config_path = tmp_path / "config.json"
    input_path = tmp_path / "requests.jsonl"
    output_path = tmp_path / "results.jsonl"
    config_path.write_text(
        json.dumps({"scores": {"critical_signal": 12.0}}),
        encoding="utf-8",
    )
    input_path.write_text(
        json.dumps(
            {
                "text": "src/a.py:1:target\nsrc/b.py:2:other",
                "tool_type": "grep",
                "budget": {
                    "keep_ratio": 0.5,
                    "no_prune_below": 0,
                    "context_window": 0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--config",
                str(config_path),
            ]
        )
        == 0
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["method"] == "execution_ast"
    assert payload["metadata"]["detected_tool_type"] == "grep"
    assert payload["metadata"]["config"]["critical_signal_score"] == 12.0
