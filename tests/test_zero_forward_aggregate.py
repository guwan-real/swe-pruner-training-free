from __future__ import annotations

import json
from pathlib import Path

from zero_forward_pruning.agent_eval.aggregate import summarize_arm


def test_aggregate_reports_zero_cost_and_recovery(tmp_path: Path) -> None:
    arm = tmp_path / "adaptive_evidence"
    trajectory_dir = arm / "instance"
    trajectory_dir.mkdir(parents=True)
    (arm / "preds.json").write_text(json.dumps({"instance": {}}), encoding="utf-8")
    (arm / "exit_code").write_text("0\n", encoding="utf-8")
    payload = {
        "info": {"exit_status": "Submitted"},
        "messages": [
            {
                "role": "assistant",
                "extra": {
                    "response": {
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 20,
                            "total_tokens": 120,
                        }
                    },
                    "actions": [
                        {
                            "command": (
                                "curl -fsS http://host.docker.internal:8124/raw/random-identifier"
                            )
                        }
                    ],
                },
            },
            {
                "role": "tool",
                "extra": {
                    "zero_forward_pruning": {
                        "status": "pruned",
                        "origin_token_cnt": 1000,
                        "left_token_cnt": 400,
                        "latency_ms": 4.5,
                        "model_input_token_cnt": 0,
                        "model_forward_count": 0,
                        "llm_token_count": 0,
                    }
                },
            },
        ],
    }
    (trajectory_dir / "instance.traj.json").write_text(json.dumps(payload), encoding="utf-8")
    summary = summarize_arm(arm)
    assert summary["agent_api_calls"] == 1
    assert summary["agent_total_tokens"] == 120
    assert summary["pruned"] == 1
    assert summary["pruner_model_forwards"] == 0
    assert summary["pruner_llm_tokens"] == 0
    assert summary["observation_retention_ratio"] == 0.4
    assert summary["recovery_actions"] == 1


def test_aggregate_reads_official_swe_pruner_fork_trajectory_shape(tmp_path: Path) -> None:
    arm = tmp_path / "adaptive_evidence"
    trajectory_dir = arm / "instance"
    trajectory_dir.mkdir(parents=True)
    payload = {
        "info": {"exit_status": "Submitted"},
        "messages": [
            {
                "role": "assistant",
                "content": (
                    "```bash\ncurl -fsS http://host.docker.internal:8124/raw/random-identifier\n```"
                ),
            },
            {
                "role": "user",
                "content": "Observation: compact",
                "pruned_stats": {
                    "status": "pruned",
                    "origin_token_cnt": 800,
                    "left_token_cnt": 200,
                    "latency_ms": 3.0,
                    "model_input_token_cnt": 0,
                    "model_forward_count": 0,
                    "llm_token_count": 0,
                },
            },
        ],
    }
    (trajectory_dir / "instance.traj.json").write_text(json.dumps(payload), encoding="utf-8")

    summary = summarize_arm(arm)

    assert summary["pruning_calls"] == 1
    assert summary["pruned"] == 1
    assert summary["observation_retention_ratio"] == 0.25
    assert summary["pruner_model_forwards"] == 0
    assert summary["recovery_actions"] == 1
