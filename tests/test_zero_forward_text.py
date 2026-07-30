from __future__ import annotations

from zero_forward_pruning.blocks import build_blocks, hard_block_indices
from zero_forward_pruning.ranking import rank_blocks
from zero_forward_pruning.text import OutputKind, classify_output


def test_output_classification_is_tool_aware() -> None:
    assert (
        classify_output(
            "def main():\n    return 1\n" * 20,
            command="sed -n '1,200p' app.py",
        )
        == OutputKind.SOURCE
    )
    assert (
        classify_output(
            "Traceback (most recent call last):\n"
            '  File "app.py", line 3, in main\n'
            "ValueError: bad\n"
        )
        == OutputKind.TRACEBACK
    )
    assert (
        classify_output("diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@")
        == OutputKind.DIFF
    )


def test_source_signatures_are_singleton_skeleton_blocks() -> None:
    text = "\n".join(
        [
            "import os",
            "",
            "def alpha():",
            "    first = 1",
            "    second = 2",
            "    return first + second",
            "",
            "def beta():",
            "    return 2",
        ]
    )
    blocks = build_blocks(text, kind=OutputKind.SOURCE, max_lines=4)
    declaration_blocks = [block for block in blocks if "structure" in block.reasons]
    assert declaration_blocks
    assert all(block.line_count == 1 for block in declaration_blocks)
    hard = hard_block_indices(blocks, OutputKind.SOURCE)
    assert all(block.index in hard for block in declaration_blocks)


def test_exact_identifier_ranks_relevant_block_first() -> None:
    text = "\n\n".join(
        [
            "def unrelated_name():\n    return 1",
            "def resolve_model(config):\n    return config['model']",
            "def another_helper():\n    return 2",
        ]
    )
    blocks = build_blocks(text, kind=OutputKind.SOURCE, max_lines=4)
    ranked = rank_blocks(blocks, "inspect resolve_model validation")
    best = blocks[ranked[0].block_index]
    assert "resolve_model" in best.text
    assert ranked[0].identifier_matches >= 1
