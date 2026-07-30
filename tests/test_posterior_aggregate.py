from __future__ import annotations

import json

from posterior_pruning.agent_eval.aggregate import summarize_arm


def test_posterior_aggregate_separates_agent_and_scorer_cost(tmp_path) -> None:
    arm = tmp_path / "single_verify"
    task = arm / "repo__issue-1"
    task.mkdir(parents=True)
    (arm / "preds.json").write_text(json.dumps({"repo__issue-1": {"instance_id": "repo__issue-1"}}))
    (task / "repo__issue-1.traj.json").write_text(
        json.dumps(
            {
                "info": {"exit_status": "Submitted"},
                "messages": [
                    {
                        "role": "assistant",
                        "content": "action",
                        "extra": {
                            "response": {
                                "usage": {
                                    "prompt_tokens": 100,
                                    "completion_tokens": 20,
                                    "total_tokens": 120,
                                }
                            }
                        },
                    },
                    {
                        "role": "user",
                        "content": "Observation: kept",
                        "posterior_pruned_stats": {
                            "status": "accepted",
                            "model_forward_count": 2,
                            "scoring_prompt_tokens": 180,
                            "original_estimated_tokens": 50,
                            "kept_estimated_tokens": 20,
                        },
                    },
                ],
            }
        )
    )

    result = summarize_arm(arm)

    assert result["agent_api_calls"] == 1
    assert result["agent_total_tokens"] == 120
    assert result["posterior_model_forwards"] == 2
    assert result["posterior_scoring_prompt_tokens"] == 180
    assert result["observation_retention_ratio"] == 0.4
    assert result["submitted"] == 1
