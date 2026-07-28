from __future__ import annotations

import pytest

from tasks.conditional_ppl import (
    ConditionalPPLConfig,
    ConditionalPPLPruner,
    HFConditionalSurprisalScorer,
    build_pruner,
)
from tf_pruning.protocol import BudgetConfig, PruningRequest


class DeterministicSurprisalScorer:
    name = "deterministic-surprisal"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, bool]] = []

    def score(
        self,
        context: str,
        continuation: str,
        *,
        first_token_only: bool = False,
    ) -> float:
        self.calls.append((context, continuation, first_token_only))
        if "CRITICAL_VALUE" in continuation:
            return 20.0 if "\n" not in continuation else 12.0
        if continuation.lstrip().startswith(("def ", "import ")):
            return 3.0
        return 1.0


def test_coarse_to_fine_surprisal_and_anchor_protection() -> None:
    text = "\n".join(
        (
            "import os",
            "def boring():",
            "    return 1",
            "def useful():",
            "    CRITICAL_VALUE = parse(payload)",
            "    return CRITICAL_VALUE",
            "noise = 0",
        )
    )
    scorer = DeterministicSurprisalScorer()
    pruner = ConditionalPPLPruner(
        scorer=scorer,
        config=ConditionalPPLConfig(
            coarse_top_fraction=0.25,
            block_max_lines=24,
            fine_context_lines=1,
            protect_structure=True,
            protect_errors=False,
        ),
    )
    request = PruningRequest(
        text=text,
        query="find parse bug",
        budget=BudgetConfig(
            keep_ratio=4 / 7,
            min_lines=4,
            no_prune_below=0,
            context_window=0,
        ),
        request_id="ppl-1",
    )

    result = pruner.prune(request)

    assert result.kept_line_numbers == (1, 2, 4, 5)
    assert result.metadata["anchor_lines"] == [1, 2, 4]
    assert result.metadata["fine_block_count"] >= 1
    assert result.metadata["fine_line_count"] < result.original_line_count
    assert result.metadata["scorer_calls"] == len(scorer.calls)
    coarse_call_count = result.metadata["block_count"]
    assert all(not first_token_only for _, _, first_token_only in scorer.calls[:coarse_call_count])
    assert all(first_token_only for _, _, first_token_only in scorer.calls[coarse_call_count:])
    critical = result.line_scores[4]
    assert critical.score > result.line_scores[6].score
    assert "fine-surprisal" in critical.reasons
    assert result.request_id == "ppl-1"


def test_explicit_metadata_anchor_is_mandatory() -> None:
    scorer = DeterministicSurprisalScorer()
    request = PruningRequest(
        text="ordinary\nordinary\nordinary\nordinary",
        budget=BudgetConfig(
            keep_ratio=0.25,
            min_lines=1,
            no_prune_below=0,
            context_window=0,
        ),
        metadata={"anchor_lines": [4]},
    )
    result = ConditionalPPLPruner(
        scorer=scorer,
        config=ConditionalPPLConfig(
            coarse_top_fraction=1.0,
            protect_structure=False,
            protect_errors=False,
        ),
    ).prune(request)
    assert result.kept_line_numbers == (4,)
    assert "protected-anchor" in result.line_scores[3].reasons


def test_no_prune_short_request_does_not_require_a_model() -> None:
    pruner = build_pruner()
    result = pruner.prune(
        PruningRequest(
            text="one\ntwo",
            budget=BudgetConfig(no_prune_below=20),
        )
    )
    assert result.kept_line_numbers == (1, 2)
    assert result.metadata["scorer_calls"] == 0


def test_hf_scorer_is_lazy_and_forces_local_files() -> None:
    scorer = HFConditionalSurprisalScorer("/models/not-loaded")
    assert scorer.is_loaded is False
    assert build_pruner({"model_path": "/models/not-loaded"}).scorer.is_loaded is False
    with pytest.raises(ValueError, match="local_files_only"):
        HFConditionalSurprisalScorer(
            "/models/not-loaded",
            local_files_only=False,
        )
