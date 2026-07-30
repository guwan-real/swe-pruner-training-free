from __future__ import annotations

from typing import Sequence

from zero_forward_pruning.blocks import (
    EvidenceBlock,
    expand_indices,
    hard_block_indices,
)
from zero_forward_pruning.methods.common import (
    BaseZeroForwardPruner,
    Selection,
    ratio_target_lines,
)
from zero_forward_pruning.protocol import PruningRequest
from zero_forward_pruning.ranking import discriminative_intent_terms, rank_blocks
from zero_forward_pruning.text import OutputKind


class IntentStructurePruner(BaseZeroForwardPruner):
    """Intent retrieval with whole-block structural neighbourhoods."""

    name = "intent_structure"

    def select(
        self,
        request: PruningRequest,
        blocks: Sequence[EvidenceBlock],
        kind: OutputKind,
    ) -> Selection:
        hard = hard_block_indices(blocks, kind)
        ranked = rank_blocks(blocks, request.intent_text)
        discriminative = discriminative_intent_terms(blocks, request.intent_text)
        seeds = set(hard)
        relevance_seeds = {
            item.block_index
            for item in ranked
            if item.block_index in hard and discriminative.intersection(item.covered_terms)
        }
        target_lines = ratio_target_lines(request, len(request.code.splitlines()))
        seed_lines = sum(block.line_count for block in blocks if block.index in seeds)
        for item in ranked:
            if seed_lines >= target_lines:
                break
            if item.score <= 0 or item.block_index in seeds:
                continue
            seeds.add(item.block_index)
            relevance_seeds.add(item.block_index)
            seed_lines += blocks[item.block_index].line_count
        radius = 2 if kind in {OutputKind.TRACEBACK, OutputKind.TEST_LOG} else 1
        expansion_seeds = relevance_seeds or {
            block.index
            for block in blocks
            if block.index in hard and "structure" not in block.reasons
        }
        selected = set(hard)
        selected.update(expand_indices(expansion_seeds, block_count=len(blocks), radius=radius))
        scores = {item.block_index: item.score for item in ranked}
        return Selection(
            block_indices=frozenset(selected),
            block_scores=scores,
            diagnostics={
                "selector": "intent-plus-structure",
                "intent_available": bool(request.intent_text.strip()),
                "discriminative_intent_terms": len(discriminative),
                "target_lines_from_contract_threshold": target_lines,
                "seed_block_count": len(seeds),
                "relevance_seed_count": len(relevance_seeds),
                "expanded_block_count": len(selected),
                "expansion_radius": radius,
            },
        )
