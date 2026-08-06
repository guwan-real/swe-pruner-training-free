from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol, Sequence

from agent_context.models import ActionEvent, ContextSignal, EvidenceDocument

WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.:/-]*|\d+")
STOPWORDS = {
    "and",
    "are",
    "bash",
    "cat",
    "cd",
    "code",
    "file",
    "for",
    "from",
    "have",
    "into",
    "not",
    "output",
    "sed",
    "that",
    "the",
    "this",
    "with",
    "your",
}


def lexical_terms(text: str) -> frozenset[str]:
    return frozenset(
        token.lower()
        for token in WORD_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS and not token.isdigit()
    )


@dataclass(frozen=True)
class SignalMatch:
    unit_scores: dict[str, float]
    matched_terms: frozenset[str]
    strategy: str

    @property
    def matched_unit_ids(self) -> frozenset[str]:
        return frozenset(unit_id for unit_id, score in self.unit_scores.items() if score > 0)


class SignalStrategy(Protocol):
    name: str

    def score(
        self,
        document: EvidenceDocument,
        signals: Sequence[ContextSignal],
    ) -> SignalMatch: ...


class SignalProvider(Protocol):
    name: str

    def from_action(self, event: ActionEvent) -> tuple[ContextSignal, ...]: ...


class PosteriorActionSignalProvider:
    name = "posterior_action"

    def from_action(self, event: ActionEvent) -> tuple[ContextSignal, ...]:
        values = (
            ("next_action.command", event.command, 2.0),
            ("next_action.focus", event.context_focus_question, 1.5),
            ("next_action.response", event.response_content[:800], 0.5),
        )
        return tuple(
            ContextSignal(provider=provider, text=text, step=event.step, weight=weight)
            for provider, text, weight in values
            if text and text.strip()
        )


class NoActionSignalProvider:
    name = "none"

    def from_action(self, event: ActionEvent) -> tuple[ContextSignal, ...]:
        del event
        return ()


class RareTermSignalStrategy:
    name = "rare_terms"

    def __init__(self, *, max_document_frequency_ratio: float = 0.1) -> None:
        if not 0.0 < max_document_frequency_ratio <= 1.0:
            raise ValueError("max_document_frequency_ratio must be in (0, 1]")
        self.max_document_frequency_ratio = max_document_frequency_ratio

    def score(
        self,
        document: EvidenceDocument,
        signals: Sequence[ContextSignal],
    ) -> SignalMatch:
        frequencies: Counter[str] = Counter()
        for unit in document.units:
            frequencies.update(unit.terms)
        cutoff = max(2, math.ceil(len(document.units) * self.max_document_frequency_ratio))
        weighted_terms: Counter[str] = Counter()
        for signal in signals:
            for term in lexical_terms(signal.text):
                if 0 < frequencies.get(term, 0) <= cutoff:
                    weighted_terms[term] += signal.weight
        unit_scores = {
            unit.id: float(sum(weighted_terms[term] for term in unit.terms))
            for unit in document.units
        }
        return SignalMatch(
            unit_scores=unit_scores,
            matched_terms=frozenset(weighted_terms),
            strategy=self.name,
        )


class AllTermsSignalStrategy:
    """Ablation strategy that skips the document-frequency safety gate."""

    name = "all_terms"

    def score(
        self,
        document: EvidenceDocument,
        signals: Sequence[ContextSignal],
    ) -> SignalMatch:
        weighted_terms: Counter[str] = Counter()
        for signal in signals:
            for term in lexical_terms(signal.text):
                weighted_terms[term] += signal.weight
        unit_scores = {
            unit.id: float(sum(weighted_terms[term] for term in unit.terms))
            for unit in document.units
        }
        return SignalMatch(
            unit_scores=unit_scores,
            matched_terms=frozenset(weighted_terms),
            strategy=self.name,
        )


class NoSignalStrategy:
    name = "none"

    def score(
        self,
        document: EvidenceDocument,
        signals: Sequence[ContextSignal],
    ) -> SignalMatch:
        del signals
        return SignalMatch(
            unit_scores={unit.id: 0.0 for unit in document.units},
            matched_terms=frozenset(),
            strategy=self.name,
        )
