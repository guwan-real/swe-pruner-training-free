from __future__ import annotations

import json
from pathlib import Path

from posterior_history_pruning.agent_eval.aggregate import summarize_run


def _write_arm(
    root: Path,
    arm: str,
    *,
    prompt_tokens: int,
    total_tokens: int,
    history_stats: dict | None,
) -> None:
    arm_root = root / "arms" / arm
    trajectory = arm_root / "instance"
    trajectory.mkdir(parents=True)
    messages: list[dict] = [
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
    if history_stats is not None:
        messages.append({"role": "user", "posterior_history_stats": history_stats})
    payload = {"info": {"exit_status": "Submitted"}, "messages": messages}
    (trajectory / "instance.traj.json").write_text(json.dumps(payload), encoding="utf-8")
    (arm_root / "started_at").write_text("2026-01-01T00:00:00+00:00\n", encoding="utf-8")
    (arm_root / "ended_at").write_text("2026-01-01T00:01:30+00:00\n", encoding="utf-8")


def test_aggregate_reports_history_savings_and_baseline_delta(tmp_path: Path) -> None:
    _write_arm(tmp_path, "baseline", prompt_tokens=1000, total_tokens=1100, history_stats=None)
    _write_arm(
        tmp_path,
        "posterior_untracked",
        prompt_tokens=1000,
        total_tokens=1100,
        history_stats={
            "status": "untracked",
            "reason": "rendered-output-boundary-not-found",
        },
    )
    _write_arm(
        tmp_path,
        "posterior_adaptive",
        prompt_tokens=800,
        total_tokens=900,
        history_stats={
            "status": "compacted",
            "reason": "posterior-action-guided",
            "token_estimator": "max-lexical-ascii4-unicode1-v2",
            "posterior_command": "rg -n resolve_model model.py",
            "origin_token_cnt": 1000,
            "left_token_cnt": 300,
            "prompt_compaction_count": 3,
            "total_prompt_tokens_saved": 2100,
        },
    )

    rows = {row["arm"]: row for row in summarize_run(tmp_path)}
    adaptive = rows["posterior_adaptive"]
    assert adaptive["agent_api_calls"] == 1
    assert adaptive["agent_api_calls_mean_per_task"] == 1
    assert adaptive["agent_api_calls_max_per_task"] == 1
    assert adaptive["agent_step_limit_hits"] == 0
    assert adaptive["history_observations_seen"] == 1
    assert adaptive["history_observations_tracked"] == 1
    assert adaptive["history_observations_untracked"] == 0
    assert adaptive["posterior_eligible_observations"] == 1
    assert adaptive["posterior_compacted_observations"] == 1
    assert adaptive["posterior_below_min_input_observations"] == 0
    assert adaptive["posterior_no_safe_reduction_observations"] == 0
    assert adaptive["history_prompt_compactions"] == 3
    assert adaptive["estimated_history_tokens_saved"] == 2100
    assert adaptive["history_token_estimators"] == "max-lexical-ascii4-unicode1-v2"
    assert adaptive["history_observation_retention_ratio"] == 0.3
    assert adaptive["agent_prompt_tokens_delta_vs_baseline"] == -200
    assert adaptive["agent_prompt_token_ratio_vs_baseline"] == 0.8
    assert adaptive["pruner_model_forwards"] == 0
    assert adaptive["pruner_llm_tokens"] == 0
    untracked = rows["posterior_untracked"]
    assert untracked["history_observations_seen"] == 1
    assert untracked["history_observations_tracked"] == 0
    assert untracked["history_observations_untracked"] == 1


def test_posterior_only_run_summarizes_without_a_baseline_directory(tmp_path: Path) -> None:
    _write_arm(
        tmp_path,
        "posterior_adaptive",
        prompt_tokens=800,
        total_tokens=900,
        history_stats={
            "status": "skipped",
            "reason": "below-min-input-tokens",
            "token_estimator": "max-lexical-ascii4-unicode1-v2",
        },
    )

    rows = summarize_run(tmp_path)

    assert len(rows) == 1
    assert rows[0]["arm"] == "posterior_adaptive"
    assert rows[0]["agent_prompt_token_ratio_vs_baseline"] is None
    assert rows[0]["posterior_below_min_input_observations"] == 1
