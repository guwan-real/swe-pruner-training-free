from __future__ import annotations

import math
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from tf_pruning.protocol import (
    PruningRequest,
    PruningResult,
    coerce_line_scores,
)
from tf_pruning.selection import render_pruned_text, select_line_numbers
from tf_pruning.text import (
    build_query,
    error_anchor_lines,
    identifiers,
    structural_anchor_lines,
    tokenize,
)

_WORD_RE = re.compile(r"\w", re.UNICODE)
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


@dataclass(frozen=True)
class IRStructuralConfig:
    """Non-learned weights and BM25 parameters.

    All values are static experiment parameters.  No fitting or online update is
    performed by :class:`IRStructuralPruner`.
    """

    bm25_weight: float = 1.0
    identifier_weight: float = 2.0
    path_weight: float = 0.75
    recent_weight: float = 1.0
    structure_weight: float = 0.35
    error_weight: float = 1.0
    window_weight: float = 0.25
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    scoring_window: int = 1
    expansion_top_k: int = 8
    preserve_structure: bool = True
    show_line_numbers: bool = True

    def __post_init__(self) -> None:
        if self.bm25_k1 <= 0:
            raise ValueError("bm25_k1 must be positive")
        if not 0.0 <= self.bm25_b <= 1.0:
            raise ValueError("bm25_b must be in [0, 1]")
        for item in fields(self):
            if item.name.endswith("_weight") and getattr(self, item.name) < 0:
                raise ValueError(f"{item.name} must be non-negative")
        if self.scoring_window < 0:
            raise ValueError("scoring_window must be non-negative")
        if self.expansion_top_k < 0:
            raise ValueError("expansion_top_k must be non-negative")

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, Any] | None,
    ) -> "IRStructuralConfig":
        if config is None:
            return cls()
        values = dict(config)
        weights = values.pop("weights", None)
        if weights is not None:
            if not isinstance(weights, Mapping):
                raise TypeError("weights must be a mapping")
            for name, value in weights.items():
                key = str(name)
                if not key.endswith("_weight"):
                    key = f"{key}_weight"
                values.setdefault(key, value)
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"unknown IR structural config keys: {unknown}")
        return cls(**values)


def _word_tokens(text: str) -> list[str]:
    """Use the shared tokenizer while dropping punctuation-only tokens."""

    return [token for token in tokenize(text) if _WORD_RE.search(token)]


def _path_terms(path: str | None) -> set[str]:
    if not path:
        return set()
    path_obj = Path(path)
    pieces: list[str] = [path_obj.name, path_obj.stem]
    pieces.extend(part for part in path_obj.parts if part not in {"/", "\\"})
    expanded: list[str] = []
    for piece in pieces:
        expanded.append(piece)
        expanded.extend(_CAMEL_BOUNDARY_RE.sub(" ", piece).split())
        expanded.extend(re.split(r"[^A-Za-z0-9_]+", piece))
    return {term.lower() for piece in expanded for term in _word_tokens(piece) if len(term) > 1}


def _overlap_fraction(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / math.sqrt(len(left) * len(right))


def _bm25_scores(
    documents: Sequence[Sequence[str]],
    query_tokens: Sequence[str],
    *,
    k1: float,
    b: float,
) -> list[float]:
    """Return Okapi BM25 scores for an in-memory line-window corpus."""

    if not documents:
        return []
    if not query_tokens:
        return [0.0] * len(documents)

    counters = [Counter(document) for document in documents]
    lengths = [sum(counter.values()) for counter in counters]
    average_length = sum(lengths) / len(lengths) if lengths else 0.0
    document_frequency: Counter[str] = Counter()
    for counter in counters:
        document_frequency.update(counter.keys())

    query_frequency = Counter(query_tokens)
    corpus_size = len(documents)
    scores: list[float] = []
    for counter, document_length in zip(counters, lengths):
        score = 0.0
        length_norm = 1.0 - b + b * document_length / average_length if average_length else 1.0
        for term, frequency_in_query in query_frequency.items():
            term_frequency = counter.get(term, 0)
            if term_frequency == 0:
                continue
            frequency = document_frequency[term]
            inverse_document_frequency = math.log(
                1.0 + (corpus_size - frequency + 0.5) / (frequency + 0.5)
            )
            saturation = term_frequency * (k1 + 1.0) / (term_frequency + k1 * length_norm)
            score += inverse_document_frequency * saturation * frequency_in_query
        scores.append(score)
    return scores


class IRStructuralPruner:
    """Rank lines with sparse retrieval signals, then preserve code structure."""

    name = "ir_structural"

    def __init__(self, config: IRStructuralConfig | None = None) -> None:
        self.config = config or IRStructuralConfig()

    def prune(self, request: PruningRequest) -> PruningResult:
        started_at = time.perf_counter()
        lines = request.lines
        line_count = len(lines)
        if not lines:
            return PruningResult(
                method=self.name,
                original_line_count=0,
                kept_line_numbers=(),
                pruned_text="",
                latency_ms=(time.perf_counter() - started_at) * 1000.0,
                metadata={"config": asdict(self.config)},
                request_id=request.request_id,
            )

        combined_query = build_query(
            request.query,
            path=request.path,
            recent_context=request.recent_context,
        )
        query_tokens = _word_tokens(combined_query)
        query_identifiers = identifiers(request.query)
        query_identifiers.update(
            identifier for item in request.recent_context for identifier in identifiers(item)
        )
        recent_tokens = {token for item in request.recent_context for token in _word_tokens(item)}
        path_tokens = _path_terms(request.path)

        documents: list[list[str]] = []
        for index in range(line_count):
            start = max(0, index - self.config.scoring_window)
            end = min(line_count, index + self.config.scoring_window + 1)
            documents.append(_word_tokens("\n".join(lines[start:end])))
        bm25 = _bm25_scores(
            documents,
            query_tokens,
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
        )

        structure = structural_anchor_lines(lines)
        errors = error_anchor_lines(lines)
        scores: list[float] = []
        reasons: dict[int, list[str]] = {}
        semantic_strength: dict[int, float] = {}
        for line_no, (line, bm25_score) in enumerate(
            zip(lines, bm25),
            start=1,
        ):
            line_ids = identifiers(line)
            line_tokens = set(_word_tokens(line))
            identifier_score = _overlap_fraction(line_ids, query_identifiers)
            path_score = _overlap_fraction(line_ids | line_tokens, path_tokens)
            recent_score = _overlap_fraction(line_ids | line_tokens, recent_tokens)

            components = {
                "bm25": self.config.bm25_weight * bm25_score,
                "identifier": self.config.identifier_weight * identifier_score,
                "path": self.config.path_weight * path_score,
                "recent": self.config.recent_weight * recent_score,
                "structure": (self.config.structure_weight if line_no in structure else 0.0),
                "error": self.config.error_weight if line_no in errors else 0.0,
            }
            score = sum(components.values())
            scores.append(score)
            semantic_strength[line_no] = sum(
                components[key] for key in ("bm25", "identifier", "path", "recent")
            )
            reasons[line_no] = [key for key, component in components.items() if component > 0.0]

        ranked_semantic = sorted(
            (line_no for line_no, score in semantic_strength.items() if score > 0.0),
            key=lambda line_no: (-semantic_strength[line_no], line_no),
        )
        semantic_seeds = ranked_semantic[: self.config.expansion_top_k]
        if self.config.window_weight > 0.0 and request.budget.context_window:
            for seed in semantic_seeds:
                seed_strength = semantic_strength[seed]
                start = max(1, seed - request.budget.context_window)
                end = min(line_count, seed + request.budget.context_window)
                for neighbour in range(start, end + 1):
                    if neighbour == seed:
                        continue
                    distance = abs(neighbour - seed)
                    scores[neighbour - 1] += (
                        self.config.window_weight * seed_strength / (distance + 1)
                    )
                    if "window" not in reasons[neighbour]:
                        reasons[neighbour].append("window")
        mandatory = structure if self.config.preserve_structure else set()
        expansion_seeds = set(semantic_seeds)
        expansion_seeds.update(mandatory)
        kept = select_line_numbers(
            scores,
            request.budget,
            mandatory=mandatory,
            expansion_seeds=expansion_seeds,
        )

        return PruningResult(
            method=self.name,
            original_line_count=line_count,
            kept_line_numbers=kept,
            pruned_text=render_pruned_text(
                lines,
                kept,
                show_line_numbers=self.config.show_line_numbers,
            ),
            line_scores=coerce_line_scores(scores, reasons),
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
            metadata={
                "config": asdict(self.config),
                "query": combined_query,
                "structural_anchor_lines": sorted(structure),
                "expansion_seed_lines": sorted(expansion_seeds),
                "training_free": True,
            },
            request_id=request.request_id,
        )
