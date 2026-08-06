from __future__ import annotations

import re
from math import ceil
from typing import Protocol

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]*|\d+|[^\w\s]", re.UNICODE)
TOKEN_ESTIMATOR_NAME = "max-lexical-ascii4-unicode1-v2"


class TokenEstimator(Protocol):
    name: str

    def estimate(self, text: str) -> int:
        """Estimate prompt tokens without depending on a model or tokenizer."""


class DeterministicTokenEstimator:
    name = TOKEN_ESTIMATOR_NAME

    def estimate(self, text: str) -> int:
        lexical_count = len(TOKEN_RE.findall(text))
        ascii_count = sum(character.isascii() for character in text)
        non_ascii_count = len(text) - ascii_count
        character_count = ceil(ascii_count / 4) + non_ascii_count
        return max(lexical_count, character_count)


DEFAULT_ESTIMATOR = DeterministicTokenEstimator()
