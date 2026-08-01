from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_eval.aggregate import convert_predictions, summarize_run
from agent_eval.config_adapter import (
    adapt_config,
    hosted_vllm_model_name,
    resolve_pruning_base_config,
    validate_pruning_contract,
)


def _base_config() -> dict:
    return {
        "agent": {
            "step_limit": 0,
            "system_template": "Use <context_focus_question>question</context_focus_question>",
            "instance_template": "Solve {{task}}",
            "format_error_template": "retry",
            "pruner": {
                "url": "http://old/prune",
                "threshold": 0.4,
            },
        },
        "environment": {"environment_class": "docker"},
        "model": {
            "model_name": "old",
            "set_cache_control": "default_end",
            "model_kwargs": {"api_base": "http://old/v1"},
        },
    }


def test_config_adapter_targets_local_vllm_without_training() -> None:
    adapted = adapt_config(
        _base_config(),
        model_id="Qwen/Qwen3.5-27B",
        api_base="http://127.0.0.1:8015/v1/",
        pruner_url="http://127.0.0.1:8113/prune",
        keep_ratio=0.35,
    )
    assert adapted["model"]["model_name"] == "hosted_vllm/Qwen/Qwen3.5-27B"
    assert adapted["model"]["model_kwargs"]["api_base"] == "http://127.0.0.1:8015/v1"
    assert adapted["model"]["model_kwargs"]["temperature"] == 0.0
    assert "set_cache_control" not in adapted["model"]
    assert adapted["agent"]["pruner"]["threshold"] == pytest.approx(0.65)
    assert adapted["agent"]["pruner"]["url"].endswith(":8113/prune")
    assert adapted["agent"]["step_limit"] == 100


def test_config_adapter_rejects_unbounded_agent_step_limit() -> None:
    with pytest.raises(ValueError, match="step_limit must be positive"):
        adapt_config(
            _base_config(),
            model_id="Qwen3.5-27B",
            api_base="http://127.0.0.1:8015/v1",
            pruner_url="http://127.0.0.1:8111/prune",
            keep_ratio=0.5,
            step_limit=0,
        )


def test_config_adapter_rejects_standard_agent_without_pruner_hook() -> None:
    with pytest.raises(ValueError, match="agent.pruner"):
        validate_pruning_contract({"agent": {"system_template": "plain"}})


def test_hosted_vllm_provider_is_not_duplicated() -> None:
    assert hosted_vllm_model_name("hosted_vllm/model") == "hosted_vllm/model"


def test_base_config_resolver_finds_one_compatible_template(tmp_path: Path) -> None:
    import yaml

    primary = tmp_path / "installed" / "swebench.yaml"
    primary.parent.mkdir()
    primary.write_text("agent:\n  system_template: plain\n", encoding="utf-8")
    template = tmp_path / "agent" / "templates" / "qwen" / "pruner.yaml"
    template.parent.mkdir(parents=True)
    template.write_text(
        yaml.safe_dump(_base_config(), sort_keys=False),
        encoding="utf-8",
    )

    resolved = resolve_pruning_base_config(
        primary,
        search_root=tmp_path / "agent",
    )
    assert resolved == template.resolve()


def test_base_config_resolver_does_not_guess_between_templates(tmp_path: Path) -> None:
    import yaml

    templates = tmp_path / "agent" / "templates"
    templates.mkdir(parents=True)
    for name in ("one.yaml", "two.yaml"):
        (templates / name).write_text(
            yaml.safe_dump(_base_config(), sort_keys=False),
            encoding="utf-8",
        )
    with pytest.raises(RuntimeError, match="MINI_SWE_BASE_CONFIG"):
        resolve_pruning_base_config(None, search_root=tmp_path / "agent")


def test_agent_summary_and_prediction_conversion(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    arm = run_root / "arms" / "ir_structural_keep50"
    arm.mkdir(parents=True)
    (arm / "preds.json").write_text(
        json.dumps(
            {
                "x": {
                    "instance_id": "x",
                    "model_name_or_path": "qwen",
                    "model_patch": "patch",
                }
            }
        ),
        encoding="utf-8",
    )
    (arm / "x.traj.json").write_text(
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
                                    "prompt_tokens": 100,
                                    "completion_tokens": 20,
                                    "total_tokens": 120,
                                }
                            }
                        },
                    },
                    {
                        "role": "user",
                        "content": "filtered",
                        "pruned_stats": {
                            "origin_token_cnt": 80,
                            "left_token_cnt": 40,
                            "model_input_token_cnt": 90,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    grade = run_root / "grade" / arm.name
    grade.mkdir(parents=True)
    (grade / "official-report.json").write_text(
        json.dumps(
            {
                "completed_instances": 1,
                "resolved_instances": 1,
            }
        ),
        encoding="utf-8",
    )

    rows = summarize_run(run_root)
    assert rows[0]["api_calls"] == 2
    assert rows[0]["api_calls_mean_per_task"] == 2
    assert rows[0]["api_calls_max_per_task"] == 2
    assert rows[0]["agent_step_limit_hits"] == 0
    assert rows[0]["observation_retention_ratio"] == 0.5
    assert rows[0]["resolve_rate"] == 1.0

    jsonl = tmp_path / "preds.jsonl"
    assert convert_predictions(arm / "preds.json", jsonl) == 1
    assert json.loads(jsonl.read_text(encoding="utf-8"))["instance_id"] == "x"
