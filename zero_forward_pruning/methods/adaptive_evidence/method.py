from __future__ import annotations

from typing import Sequence

from zero_forward_pruning.blocks import (
    EvidenceBlock,
    expand_indices,
    hard_block_indices,
)
from zero_forward_pruning.methods.common import BaseZeroForwardPruner, Selection
from zero_forward_pruning.protocol import PruningRequest
from zero_forward_pruning.ranking import discriminative_intent_terms, rank_blocks
from zero_forward_pruning.text import OutputKind, terms

SOFT_BLOCK_CAP = {
    OutputKind.SOURCE: 12,
    OutputKind.TRACEBACK: 12,
    OutputKind.TEST_LOG: 12,
    OutputKind.SEARCH: 16,
    OutputKind.TREE: 10,
    OutputKind.GENERIC: 8,
    OutputKind.DIFF: 0,
}


class AdaptiveEvidencePruner(BaseZeroForwardPruner):
    """Recommended coverage-driven method; it does not search keep ratios."""

    name = "adaptive_evidence"

    def select(
        self,
        request: PruningRequest,
        blocks: Sequence[EvidenceBlock],
        kind: OutputKind,
    ) -> Selection:
        hard = hard_block_indices(blocks, kind)
        ranked = rank_blocks(blocks, request.intent_text)
        discriminative = discriminative_intent_terms(blocks, request.intent_text)
        selected = set(hard)
        relevance_seeds = {
            item.block_index
            for item in ranked
            if item.block_index in hard and discriminative.intersection(item.covered_terms)
        }
        query_terms = set(terms(request.intent_text))
        present_terms = {term for block in blocks for term in query_terms.intersection(block.terms)}
        covered = {
            term
            for block in blocks
            if block.index in selected
            for term in present_terms.intersection(block.terms)
        }
        positive = [item for item in ranked if item.score > 0]
        soft_added = 0
        cap = SOFT_BLOCK_CAP[kind]
        # First guarantee that every intent term found in the observation has
        # at least one evidence block.
        while covered != present_terms and soft_added < cap:
            candidates = [
                item
                for item in positive
                if item.block_index not in selected and set(item.covered_terms).difference(covered)
            ]
            if not candidates:
                break
            item = max(
                candidates,
                key=lambda candidate: (
                    len(set(candidate.covered_terms).difference(covered)),
                    candidate.identifier_matches,
                    candidate.score,
                    -candidate.block_index,
                ),
            )
            selected.add(item.block_index)
            relevance_seeds.add(item.block_index)
            covered.update(item.covered_terms)
            soft_added += 1
        # Exact identifiers and the strongest remaining evidence are useful
        # even after lexical coverage is complete.  This is a fixed one-pass
        # policy, not a model-evaluated threshold sweep.
        for item in positive:
            if soft_added >= cap:
                break
            if item.block_index in selected:
                continue
            if item.identifier_matches == 0 and soft_added >= 2:
                continue
            selected.add(item.block_index)
            relevance_seeds.add(item.block_index)
            soft_added += 1
        if kind in {OutputKind.SOURCE, OutputKind.TRACEBACK, OutputKind.TEST_LOG}:
            radius = 2 if kind != OutputKind.SOURCE else 1
            expansion_seeds = relevance_seeds or {
                block.index
                for block in blocks
                if block.index in hard and "structure" not in block.reasons
            }
            selected.update(expand_indices(expansion_seeds, block_count=len(blocks), radius=radius))
        else:
            radius = 0
        scores = {item.block_index: item.score for item in ranked}
        return Selection(
            block_indices=frozenset(selected),
            block_scores=scores,
            diagnostics={
                "selector": "coverage-driven-adaptive-evidence",
                "contract_threshold_ignored": True,
                "intent_available": bool(request.intent_text.strip()),
                "discriminative_intent_terms": len(discriminative),
                "intent_terms_present": len(present_terms),
                "intent_terms_covered": len(covered),
                "hard_block_count": len(hard),
                "soft_blocks_added": soft_added,
                "soft_block_cap": cap,
                "expansion_radius": radius,
            },
        )
