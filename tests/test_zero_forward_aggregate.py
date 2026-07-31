from __future__ import annotations

import json
from pathlib import Path

from zero_forward_pruning.agent_eval.aggregate import summarize_arm, summarize_run


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
    assert summary["server_requests"] == 1
    assert summary["server_pruned"] == 1
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


def test_aggregate_separates_client_server_and_recovery_guard_counts(tmp_path: Path) -> None:
    arm = tmp_path / "adaptive_evidence"
    trajectory_dir = arm / "instance"
    trajectory_dir.mkdir(parents=True)
    payload = {
        "messages": [
            {
                "role": "user",
                "pruned_stats": {
                    "method": "client",
                    "status": "skipped",
                    "origin_token_cnt": 100,
                    "left_token_cnt": 100,
                    "diagnostics": {"reason": "below-client-min-chars"},
                },
            },
            {
                "role": "user",
                "pruned_stats": {
                    "method": "adaptive_evidence",
                    "status": "pruned",
                    "origin_token_cnt": 1000,
                    "left_token_cnt": 400,
                    "diagnostics": {"reason": "cost-gate-passed"},
                },
            },
            {
                "role": "user",
                "pruned_stats": {
                    "method": "recovery_guard",
                    "status": "guarded",
                    "origin_token_cnt": 800,
                    "left_token_cnt": 80,
                    "diagnostics": {"reason": "unbounded-recovery-output-withheld"},
                },
            },
        ]
    }
    (trajectory_dir / "instance.traj.json").write_text(json.dumps(payload), encoding="utf-8")

    summary = summarize_arm(arm)

    assert summary["pruning_attempts"] == 3
    assert summary["client_skips"] == 1
    assert summary["server_requests"] == 1
    assert summary["server_pruned"] == 1
    assert summary["server_skipped"] == 0
    assert summary["recovery_guarded"] == 1
    assert summary["server_observation_retention_ratio"] == 0.4
    assert summary["effective_pruned_retention_ratio"] == 0.4
    assert summary["observation_retention_ratio"] == 580 / 1900
    assert summary["pruning_reasons"]["below-client-min-chars"] == 1
    assert summary["pruning_reasons"]["unbounded-recovery-output-withheld"] == 1


def test_run_summary_compares_equal_task_arms_to_baseline(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    for arm, prompt_tokens, total_tokens, wall_seconds in (
        ("baseline", 1000, 1100, 100),
        ("adaptive_evidence", 800, 900, 90),
    ):
        arm_root = run_root / "arms" / arm
        trajectory_root = arm_root / "instance"
        trajectory_root.mkdir(parents=True)
        payload = {
            "messages": [
                {
                    "role": "assistant",
                    "extra": {
                        "response": {
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": total_tokens - prompt_tokens,
                                "total_tokens": total_tokens,
                            }
                        }
                    },
                }
            ]
        }
        (trajectory_root / "instance.traj.json").write_text(json.dumps(payload), encoding="utf-8")
        (arm_root / "started_at").write_text("2026-01-01T00:00:00+00:00\n", encoding="utf-8")
        (arm_root / "ended_at").write_text(
            f"2026-01-01T00:01:{wall_seconds - 60:02d}+00:00\n"
            if wall_seconds < 120
            else "2026-01-01T00:02:00+00:00\n",
            encoding="utf-8",
        )

    rows = {row["arm"]: row for row in summarize_run(run_root)}
    baseline = rows["baseline"]
    adaptive = rows["adaptive_evidence"]
    assert baseline["agent_prompt_tokens_delta_vs_baseline"] == 0
    assert baseline["agent_prompt_token_ratio_vs_baseline"] == 1.0
    assert adaptive["agent_prompt_tokens_delta_vs_baseline"] == -200
    assert adaptive["agent_prompt_token_ratio_vs_baseline"] == 0.8
    assert adaptive["agent_total_tokens_delta_vs_baseline"] == -200
    assert adaptive["wall_time_seconds_delta_vs_baseline"] == -10
    assert adaptive["wall_time_ratio_vs_baseline"] == 0.9
