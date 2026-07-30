from __future__ import annotations

import pytest

from posterior_pruning.scoring import ScoringError, VLLMActionScorer, VLLMScorerConfig


def test_vllm_scorer_uses_exact_chat_template_boundary_and_actual_token_logprobs() -> None:
    calls: list[tuple[str, str, object]] = []

    def transport(method, url, payload, timeout):
        calls.append((method, url, payload))
        if url.endswith("/tokenize"):
            assert payload["add_generation_prompt"] is True
            assert payload["messages"][-1]["role"] == "user"
            return {"tokens": [10, 11]}
        if url.endswith("/chat/completions"):
            assert payload["add_generation_prompt"] is False
            assert payload["messages"][-1] == {
                "role": "assistant",
                "content": "fixed action",
            }
            assert payload["prompt_logprobs"] == 1
            assert payload["return_token_ids"] is True
            return {
                "prompt_token_ids": [10, 11, 20, 21],
                "prompt_logprobs": [
                    None,
                    {"11": {"logprob": -0.1}},
                    {"20": {"logprob": -0.2}},
                    {"21": {"logprob": -0.4}},
                ],
            }
        raise AssertionError(url)

    scorer = VLLMActionScorer(
        VLLMScorerConfig(model="Qwen3.5-27B"),
        transport=transport,
    )

    score = scorer.score([{"role": "user", "content": "history"}], "fixed action")

    assert score.sum_logprob == pytest.approx(-0.6)
    assert score.mean_logprob == pytest.approx(-0.3)
    assert score.target_tokens == 2
    assert score.prompt_tokens == 4
    assert len(calls) == 2


def test_vllm_scorer_rejects_a_chat_template_mismatch() -> None:
    def transport(method, url, payload, timeout):
        if url.endswith("/tokenize"):
            return {"tokens": [1, 2]}
        return {
            "prompt_token_ids": [1, 9, 3],
            "prompt_logprobs": [
                None,
                {"9": {"logprob": -0.1}},
                {"3": {"logprob": -0.1}},
            ],
        }

    scorer = VLLMActionScorer(
        VLLMScorerConfig(model="Qwen3.5-27B"),
        transport=transport,
    )

    with pytest.raises(ScoringError, match="chat-template prefix mismatch"):
        scorer.score([{"role": "user", "content": "history"}], "action")
