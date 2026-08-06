from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_context.agent_eval.aggregate import convert_predictions, summarize_run


def _metrics(
    *,
    prompt_tokens: int,
    prefix_queries: int,
    prefix_hits: int,
    requests: int,
    ttft: float,
    prefill: float,
    e2e: float,
) -> str:
    return "\n".join(
        (
            f'vllm:prompt_tokens_total{{model_name="qwen"}} {prompt_tokens}',
            f'vllm:prefix_cache_queries_total{{model_name="qwen"}} {prefix_queries}',
            f'vllm:prefix_cache_hits_total{{model_name="qwen"}} {prefix_hits}',
            f'vllm:request_success_total{{finished_reason="stop"}} {requests}',
            f"vllm:time_to_first_token_seconds_sum {ttft}",
            f"vllm:time_to_first_token_seconds_count {requests}",
            f"vllm:request_prefill_time_seconds_sum {prefill}",
            f"vllm:request_prefill_time_seconds_count {requests}",
            f"vllm:e2e_request_latency_seconds_sum {e2e}",
            f"vllm:e2e_request_latency_seconds_count {requests}",
            "",
        )
    )


def _write_arm(
    root: Path,
    arm: str,
    *,
    prompt_tokens: int,
    cached_tokens: int,
    full_tokens: int,
    selected_tokens: int,
    overflow: int,
    view_switches: int,
    wall_seconds: int,
    metrics_before: str,
    metrics_after: str,
    exit_status: str = "Submitted",
) -> None:
    arm_root = root / "arms" / arm
    trajectory_root = arm_root / "instance"
    trajectory_root.mkdir(parents=True)
    manifest = {
        "full_observation_tokens": full_tokens,
        "selected_observation_tokens": selected_tokens,
        "estimated_tokens_saved": full_tokens - selected_tokens,
        "budget_overflow_tokens": overflow,
        "context_view_switches": view_switches,
        "entries": [
            {
                "selected_level": "focused",
                "kind": "source",
                "selection_changed": bool(view_switches),
            },
            {
                "selected_level": "full",
                "kind": "diff",
                "selection_changed": False,
            },
        ],
    }
    payload = {
        "info": {"exit_status": exit_status, "model_stats": {"api_calls": 1}},
        "messages": [
            {
                "role": "assistant",
                "extra": {
                    "response": {
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": 20,
                            "total_tokens": prompt_tokens + 20,
                            "prompt_tokens_details": {"cached_tokens": cached_tokens},
                        }
                    }
                },
                "agent_context_manifest": manifest,
            },
            {
                "role": "user",
                "agent_context_stats": {"status": "tracked", "reason": "tracked"},
                "agent_context_report": {"committed_observations": 1},
            },
        ],
    }
    (trajectory_root / "instance.traj.json").write_text(json.dumps(payload), encoding="utf-8")
    (arm_root / "preds.json").write_text(
        json.dumps({"instance": {"instance_id": "instance", "model_patch": "patch"}}),
        encoding="utf-8",
    )
    (arm_root / "started_at").write_text("2026-01-01T00:00:00Z\n", encoding="utf-8")
    (arm_root / "ended_at").write_text(
        f"2026-01-01T00:{wall_seconds // 60:02d}:{wall_seconds % 60:02d}Z\n",
        encoding="utf-8",
    )
    (arm_root / "exit_code").write_text("0\n", encoding="utf-8")
    (arm_root / "vllm_metrics_before.prom").write_text(metrics_before, encoding="utf-8")
    (arm_root / "vllm_metrics_after.prom").write_text(metrics_after, encoding="utf-8")


def test_aggregate_reports_context_cache_and_latency_metrics(tmp_path: Path) -> None:
    before = _metrics(
        prompt_tokens=1000,
        prefix_queries=100,
        prefix_hits=80,
        requests=10,
        ttft=20,
        prefill=10,
        e2e=40,
    )
    _write_arm(
        tmp_path,
        "R",
        prompt_tokens=1000,
        cached_tokens=800,
        full_tokens=100,
        selected_tokens=80,
        overflow=0,
        view_switches=0,
        wall_seconds=60,
        metrics_before=before,
        metrics_after=_metrics(
            prompt_tokens=2000,
            prefix_queries=200,
            prefix_hits=160,
            requests=11,
            ttft=22,
            prefill=12,
            e2e=44,
        ),
    )
    _write_arm(
        tmp_path,
        "C",
        prompt_tokens=800,
        cached_tokens=200,
        full_tokens=100,
        selected_tokens=60,
        overflow=10,
        view_switches=2,
        wall_seconds=90,
        metrics_before=before,
        metrics_after=_metrics(
            prompt_tokens=1800,
            prefix_queries=200,
            prefix_hits=105,
            requests=11,
            ttft=26,
            prefill=16,
            e2e=52,
        ),
    )

    rows = {row["arm"]: row for row in summarize_run(tmp_path)}
    reference = rows["R"]
    dynamic = rows["C"]

    assert reference["context_history_retention_ratio"] == 0.8
    assert reference["context_estimated_tokens_saved"] == 20
    assert reference["context_compact_prompt_entries"] == 1
    assert reference["context_committed_observations"] == 1
    assert reference["agent_uncached_prompt_tokens"] == 200
    assert reference["prompt_cache_hit_ratio"] == 0.8
    assert reference["vllm_metrics_available"] is True
    assert reference["vllm_metrics_isolated"] is True
    assert reference["vllm_prefix_cache_hit_ratio"] == 0.8
    assert reference["vllm_ttft_seconds_mean"] == 2.0
    assert reference["vllm_prefill_seconds_total"] == 2.0
    assert dynamic["context_budget_overflow_tokens"] == 10
    assert dynamic["context_budget_overflow_prompts"] == 1
    assert dynamic["context_view_switches"] == 2
    assert dynamic["agent_prompt_token_ratio_vs_reference"] == 0.8
    assert dynamic["wall_time_ratio_vs_reference"] == 1.5
    assert dynamic["vllm_prefix_cache_hit_ratio"] == 0.25
    assert dynamic["vllm_prefix_cache_hit_ratio_delta_vs_reference"] == pytest.approx(-0.55)
    assert dynamic["vllm_prefill_time_ratio_vs_reference"] == 3.0


def test_aggregate_rejects_vllm_deltas_contaminated_by_other_clients(tmp_path: Path) -> None:
    before = _metrics(
        prompt_tokens=0,
        prefix_queries=0,
        prefix_hits=0,
        requests=0,
        ttft=0,
        prefill=0,
        e2e=0,
    )
    _write_arm(
        tmp_path,
        "R",
        prompt_tokens=10,
        cached_tokens=0,
        full_tokens=10,
        selected_tokens=10,
        overflow=0,
        view_switches=0,
        wall_seconds=10,
        metrics_before=before,
        metrics_after=_metrics(
            prompt_tokens=10,
            prefix_queries=10,
            prefix_hits=5,
            requests=1,
            ttft=1,
            prefill=1,
            e2e=2,
        ),
    )
    _write_arm(
        tmp_path,
        "C",
        prompt_tokens=10,
        cached_tokens=0,
        full_tokens=10,
        selected_tokens=10,
        overflow=0,
        view_switches=0,
        wall_seconds=10,
        metrics_before=before,
        metrics_after=_metrics(
            prompt_tokens=20,
            prefix_queries=20,
            prefix_hits=10,
            requests=2,
            ttft=2,
            prefill=2,
            e2e=4,
        ),
    )

    rows = {row["arm"]: row for row in summarize_run(tmp_path)}
    assert rows["R"]["vllm_metrics_isolated"] is True
    assert rows["C"]["vllm_metrics_available"] is True
    assert rows["C"]["vllm_metrics_isolated"] is False
    assert rows["C"]["vllm_prefix_cache_hit_ratio_delta_vs_reference"] is None
    assert rows["C"]["vllm_prefill_time_ratio_vs_reference"] is None


def test_aggregate_handles_missing_cache_telemetry_and_converts_predictions(
    tmp_path: Path,
) -> None:
    arm = tmp_path / "arms" / "R"
    trajectory = arm / "instance"
    trajectory.mkdir(parents=True)
    (trajectory / "instance.traj.json").write_text(
        json.dumps(
            {
                "info": {"model_stats": {"api_calls": 1}},
                "messages": [
                    {
                        "role": "assistant",
                        "extra": {
                            "response": {"usage": {"prompt_tokens": 10, "completion_tokens": 1}}
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (arm / "preds.json").write_text(
        json.dumps({"instance": {"instance_id": "instance", "model_patch": "patch"}}),
        encoding="utf-8",
    )

    row = summarize_run(tmp_path)[0]
    assert row["cache_telemetry_complete"] is False
    assert row["agent_cached_prompt_tokens"] is None
    assert row["vllm_metrics_available"] is False

    output = tmp_path / "preds.jsonl"
    assert convert_predictions(arm / "preds.json", output) == 1
    assert json.loads(output.read_text(encoding="utf-8"))["instance_id"] == "instance"


def test_aggregate_rejects_partial_usage_for_cache_ratios(tmp_path: Path) -> None:
    arm = tmp_path / "arms" / "R"
    trajectory = arm / "instance"
    trajectory.mkdir(parents=True)
    (trajectory / "instance.traj.json").write_text(
        json.dumps(
            {
                "info": {
                    "exit_status": "Submitted",
                    "model_stats": {"api_calls": 2},
                },
                "messages": [
                    {
                        "role": "assistant",
                        "extra": {
                            "response": {
                                "usage": {
                                    "prompt_tokens": 10,
                                    "completion_tokens": 1,
                                    "prompt_tokens_details": {"cached_tokens": 8},
                                }
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (arm / "preds.json").write_text(
        json.dumps({"instance": {"instance_id": "instance"}}), encoding="utf-8"
    )

    row = summarize_run(tmp_path)[0]
    assert row["agent_api_calls"] == 2
    assert row["cache_telemetry_calls"] == 1
    assert row["usage_telemetry_complete"] is False
    assert row["cache_telemetry_complete"] is False
    assert row["prompt_cache_hit_ratio"] is None


def test_aggregate_rejects_a_whole_trajectory_without_usage(tmp_path: Path) -> None:
    metrics = _metrics(
        prompt_tokens=0,
        prefix_queries=0,
        prefix_hits=0,
        requests=0,
        ttft=0,
        prefill=0,
        e2e=0,
    )
    _write_arm(
        tmp_path,
        "R",
        prompt_tokens=10,
        cached_tokens=8,
        full_tokens=10,
        selected_tokens=10,
        overflow=0,
        view_switches=0,
        wall_seconds=10,
        metrics_before=metrics,
        metrics_after=metrics,
    )
    arm = tmp_path / "arms" / "R"
    second = arm / "second" / "second.traj.json"
    second.parent.mkdir()
    second.write_text(
        json.dumps(
            {
                "info": {"exit_status": "Submitted"},
                "messages": [],
            }
        ),
        encoding="utf-8",
    )
    predictions = json.loads((arm / "preds.json").read_text(encoding="utf-8"))
    predictions["second"] = {"instance_id": "second", "model_patch": "patch"}
    (arm / "preds.json").write_text(json.dumps(predictions), encoding="utf-8")

    row = summarize_run(tmp_path)[0]
    assert row["all_tasks_submitted"] is True
    assert row["usage_telemetry_complete"] is False
    assert row["cache_telemetry_complete"] is False
    assert row["prompt_cache_hit_ratio"] is None


def test_aggregate_does_not_compare_failed_or_incomplete_arms(tmp_path: Path) -> None:
    before = _metrics(
        prompt_tokens=0,
        prefix_queries=0,
        prefix_hits=0,
        requests=0,
        ttft=0,
        prefill=0,
        e2e=0,
    )
    _write_arm(
        tmp_path,
        "R",
        prompt_tokens=10,
        cached_tokens=0,
        full_tokens=10,
        selected_tokens=10,
        overflow=0,
        view_switches=0,
        wall_seconds=10,
        metrics_before=before,
        metrics_after=before,
    )
    _write_arm(
        tmp_path,
        "failed",
        prompt_tokens=10,
        cached_tokens=0,
        full_tokens=10,
        selected_tokens=5,
        overflow=0,
        view_switches=0,
        wall_seconds=5,
        metrics_before=before,
        metrics_after=before,
    )
    _write_arm(
        tmp_path,
        "running",
        prompt_tokens=10,
        cached_tokens=0,
        full_tokens=10,
        selected_tokens=5,
        overflow=0,
        view_switches=0,
        wall_seconds=5,
        metrics_before=before,
        metrics_after=before,
    )
    (tmp_path / "arms" / "failed" / "exit_code").write_text("7\n", encoding="utf-8")
    (tmp_path / "arms" / "running" / "ended_at").unlink()
    (tmp_path / "arms" / "running" / "exit_code").unlink()

    rows = {row["arm"]: row for row in summarize_run(tmp_path)}
    assert rows["R"]["runner_completed"] is True
    assert rows["failed"]["runner_completed"] is True
    assert rows["failed"]["agent_prompt_token_ratio_vs_reference"] is None
    assert rows["failed"]["wall_time_ratio_vs_reference"] is None
    assert rows["running"]["runner_completed"] is False
    assert rows["running"]["agent_prompt_token_ratio_vs_reference"] is None


def test_aggregate_does_not_compare_non_submitted_task_outcomes(tmp_path: Path) -> None:
    metrics = _metrics(
        prompt_tokens=0,
        prefix_queries=0,
        prefix_hits=0,
        requests=0,
        ttft=0,
        prefill=0,
        e2e=0,
    )
    _write_arm(
        tmp_path,
        "R",
        prompt_tokens=10,
        cached_tokens=0,
        full_tokens=10,
        selected_tokens=10,
        overflow=0,
        view_switches=0,
        wall_seconds=10,
        metrics_before=metrics,
        metrics_after=metrics,
    )
    _write_arm(
        tmp_path,
        "C",
        prompt_tokens=5,
        cached_tokens=0,
        full_tokens=10,
        selected_tokens=5,
        overflow=0,
        view_switches=0,
        wall_seconds=10,
        metrics_before=metrics,
        metrics_after=metrics,
        exit_status="LimitsExceeded",
    )

    rows = {row["arm"]: row for row in summarize_run(tmp_path)}
    assert rows["R"]["all_tasks_submitted"] is True
    assert rows["C"]["all_tasks_submitted"] is False
    assert rows["C"]["agent_prompt_token_ratio_vs_reference"] is None
    assert rows["C"]["wall_time_ratio_vs_reference"] is None
    assert rows["C"]["vllm_prefix_cache_hit_ratio_delta_vs_reference"] is None
