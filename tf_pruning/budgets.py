from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .protocol import BudgetConfig


@dataclass(frozen=True)
class LengthBand:
    max_lines: int | None
    keep_ratio: float

    def __post_init__(self) -> None:
        if self.max_lines is not None and self.max_lines < 0:
            raise ValueError("max_lines must be non-negative")
        if not 0.0 < self.keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1]")


@dataclass(frozen=True)
class LengthAwareBudget:
    bands: tuple[LengthBand, ...]
    min_lines: int = 1
    hard_max_lines: int | None = None
    no_prune_below: int = 20
    context_window: int = 1

    def __post_init__(self) -> None:
        if not self.bands:
            raise ValueError("at least one length band is required")
        finite = [band.max_lines for band in self.bands if band.max_lines is not None]
        if finite != sorted(finite) or len(finite) != len(set(finite)):
            raise ValueError("finite band boundaries must be unique and sorted")
        if self.bands[-1].max_lines is not None:
            raise ValueError("the final length band must be open-ended")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LengthAwareBudget":
        raw_bands = payload.get("bands")
        if not isinstance(raw_bands, Sequence) or isinstance(raw_bands, (str, bytes)):
            raise ValueError("bands must be a list")
        bands = tuple(
            LengthBand(
                max_lines=(
                    None if dict(item).get("max_lines") is None else int(dict(item)["max_lines"])
                ),
                keep_ratio=float(dict(item)["keep_ratio"]),
            )
            for item in raw_bands
        )
        return cls(
            bands=bands,
            min_lines=int(payload.get("min_lines", 1)),
            hard_max_lines=(
                None if payload.get("hard_max_lines") is None else int(payload["hard_max_lines"])
            ),
            no_prune_below=int(payload.get("no_prune_below", 20)),
            context_window=int(payload.get("context_window", 1)),
        )

    def for_line_count(self, line_count: int) -> BudgetConfig:
        ratio = self.bands[-1].keep_ratio
        for band in self.bands:
            if band.max_lines is None or line_count <= band.max_lines:
                ratio = band.keep_ratio
                break
        return BudgetConfig(
            keep_ratio=ratio,
            min_lines=self.min_lines,
            max_lines=self.hard_max_lines,
            no_prune_below=self.no_prune_below,
            context_window=self.context_window,
        )


DEFAULT_LENGTH_AWARE_BUDGET = LengthAwareBudget(
    bands=(
        LengthBand(max_lines=20, keep_ratio=1.0),
        LengthBand(max_lines=80, keep_ratio=0.7),
        LengthBand(max_lines=200, keep_ratio=0.45),
        LengthBand(max_lines=None, keep_ratio=0.3),
    )
)
