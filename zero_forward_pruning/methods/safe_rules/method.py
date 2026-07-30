from __future__ import annotations

from typing import Sequence

from zero_forward_pruning.blocks import (
    EvidenceBlock,
    expand_indices,
    hard_block_indices,
)
from zero_forward_pruning.methods.common import BaseZeroForwardPruner, Selection
from zero_forward_pruning.protocol import PruningRequest
from zero_forward_pruning.text import OutputKind


class SafeRulesPruner(BaseZeroForwardPruner):
    """Tool-specific hard evidence and structural skeleton only."""

    name = "safe_rules"

    def select(
        self,
        request: PruningRequest,
        blocks: Sequence[EvidenceBlock],
        kind: OutputKind,
    ) -> Selection:
        del request
        hard = hard_block_indices(blocks, kind)
        semantic_hard = {block.index for block in blocks if block.reasons}
        if kind in {OutputKind.GENERIC, OutputKind.TREE} and not semantic_hard:
            return Selection(
                block_indices=frozenset(range(len(blocks))),
                diagnostics={
                    "selector": "hard-evidence-only",
                    "reason": "no-tool-specific-hard-evidence",
                },
            )
        if kind in {OutputKind.TRACEBACK, OutputKind.TEST_LOG}:
            selected = expand_indices(hard, block_count=len(blocks), radius=1)
        elif kind == OutputKind.SOURCE:
            selected = set(hard)
        elif kind == OutputKind.SEARCH:
            selected = set(hard)
        else:
            selected = expand_indices(hard, block_count=len(blocks), radius=1)
        scores = {index: 1.0 for index in selected}
        return Selection(
            block_indices=frozenset(selected),
            block_scores=scores,
            diagnostics={
                "selector": "hard-evidence-only",
                "hard_block_count": len(hard),
            },
        )
