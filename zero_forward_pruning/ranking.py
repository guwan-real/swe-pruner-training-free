from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from zero_forward_pruning.blocks import EvidenceBlock, document_frequency
from zero_forward_pruning.text import identifiers, terms


@dataclass(frozen=True)
class RankedBlock:
    block_index: int
    score: float
    bm25_score: float
    identifier_matches: int
    covered_terms: tuple[str, ...]


def _bm25_scores(blocks: Sequence[EvidenceBlock], query_terms: tuple[str, ...]) -> dict[int, float]:
    if not blocks or not query_terms:
        return {block.index: 0.0 for block in blocks}
    frequencies = document_frequency(blocks)
    average_length = sum(len(block.terms) for block in blocks) / len(blocks)
    average_length = max(average_length, 1.0)
    query_counts = Counter(query_terms)
    scores: dict[int, float] = {}
    for block in blocks:
        counts = Counter(block.terms)
        score = 0.0
        for term, query_frequency in query_counts.items():
            frequency = counts[term]
            if frequency == 0:
                continue
            inverse_frequency = math.log(
                1.0 + (len(blocks) - frequencies[term] + 0.5) / (frequencies[term] + 0.5)
            )
            denominator = frequency + 1.2 * (0.25 + 0.75 * len(block.terms) / average_length)
            score += inverse_frequency * frequency * 2.2 / denominator * query_frequency
        scores[block.index] = score
    return scores


def _ranks(values: dict[int, float]) -> dict[int, int]:
    ordered = sorted(values, key=lambda index: (-values[index], index))
    return {index: rank for rank, index in enumerate(ordered, start=1)}


def rank_blocks(blocks: Sequence[EvidenceBlock], intent_text: str) -> list[RankedBlock]:
    """Fuse lexical and exact-code rankings without fitted parameters."""

    query_terms = terms(intent_text)
    query_identifiers = identifiers(intent_text)
    bm25 = _bm25_scores(blocks, query_terms)
    exact = {
        block.index: float(len(query_identifiers.intersection(block.identifiers)))
        for block in blocks
    }
    evidence = {block.index: float(len(block.reasons)) for block in blocks}
    bm25_ranks = _ranks(bm25)
    exact_ranks = _ranks(exact)
    evidence_ranks = _ranks(evidence)
    query_term_set = set(query_terms)
    ranked: list[RankedBlock] = []
    for block in blocks:
        # Reciprocal-rank fusion avoids training or hand-fitting a linear classifier.
        score = 0.0
        if bm25[block.index] > 0:
            score += 1.0 / (60 + bm25_ranks[block.index])
        if exact[block.index] > 0:
            score += 1.0 / (60 + exact_ranks[block.index])
        if evidence[block.index] > 0:
            score += 1.0 / (60 + evidence_ranks[block.index])
        covered = tuple(sorted(query_term_set.intersection(block.terms)))
        ranked.append(
            RankedBlock(
                block_index=block.index,
                score=score,
                bm25_score=bm25[block.index],
                identifier_matches=int(exact[block.index]),
                covered_terms=covered,
            )
        )
    return sorted(ranked, key=lambda item: (-item.score, item.block_index))


def discriminative_intent_terms(
    blocks: Sequence[EvidenceBlock],
    intent_text: str,
) -> set[str]:
    """Return intent terms rare enough to identify a local evidence region."""

    query_terms = set(terms(intent_text))
    frequencies = document_frequency(blocks)
    cutoff = max(3, math.ceil(len(blocks) * 0.1))
    return {term for term in query_terms if 0 < frequencies.get(term, 0) <= cutoff}
