from __future__ import annotations

import pytest

from zero_forward_pruning.protocol import PruningRequest, PruningResult


def test_official_request_contract_and_optional_context() -> None:
    request = PruningRequest.from_dict(
        {
            "query": "find resolver",
            "code": "def resolver():\n    pass",
            "threshold": 0.4,
            "recent_context": ["first", "second"],
            "metadata": {"source": "test"},
        }
    )
    assert request.query == "find resolver"
    assert request.recent_context == "first\nsecond"
    assert request.threshold == 0.4


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_request_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        PruningRequest.from_dict({"query": "", "code": "x", "threshold": threshold})


def test_result_enforces_zero_forward_invariants() -> None:
    with pytest.raises(ValueError, match="model_forward_count"):
        PruningResult(
            pruned_code="x",
            origin_token_cnt=1,
            left_token_cnt=1,
            model_input_token_cnt=0,
            model_forward_count=1,
            method="bad",
            status="pruned",
            original_line_count=1,
            kept_line_count=1,
            retention_ratio=1.0,
            latency_ms=0.1,
        )
