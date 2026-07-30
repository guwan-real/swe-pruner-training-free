from __future__ import annotations

from typing import Sequence

from zero_forward_pruning.blocks import EvidenceBlock, hard_block_indices
from zero_forward_pruning.methods.common import (
    BaseZeroForwardPruner,
    Selection,
    ratio_target_lines,
)
from zero_forward_pruning.protocol import PruningRequest
from zero_forward_pruning.ranking import rank_blocks
from zero_forward_pruning.text import OutputKind


class IntentIRPruner(BaseZeroForwardPruner):
    """Query-conditioned sparse retrieval without structural expansion."""

    name = "intent_ir"

    def select(
        self,
        request: PruningRequest,
        blocks: Sequence[EvidenceBlock],
        kind: OutputKind,
    ) -> Selection:
        hard = hard_block_indices(blocks, kind)
        ranked = rank_blocks(blocks, request.intent_text)
        selected = set(hard)
        target_lines = ratio_target_lines(request, len(request.code.splitlines()))
        line_count = sum(block.line_count for block in blocks if block.index in selected)
        for item in ranked:
            if line_count >= target_lines:
                break
            if item.score <= 0 or item.block_index in selected:
                continue
            selected.add(item.block_index)
            line_count += blocks[item.block_index].line_count
        scores = {item.block_index: item.score for item in ranked}
        return Selection(
            block_indices=frozenset(selected),
            block_scores=scores,
            diagnostics={
                "selector": "intent-ir",
                "intent_available": bool(request.intent_text.strip()),
                "target_lines_from_contract_threshold": target_lines,
                "hard_block_count": len(hard),
            },
        )
