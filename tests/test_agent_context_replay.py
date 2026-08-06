from __future__ import annotations

import ast
from pathlib import Path

from agent_context.cli import main
from agent_context.config import AgentContextConfig
from agent_context.replay import replay_trace

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_structured_trace_replay_materializes_manifests_without_model_calls() -> None:
    source = "\n".join(
        ["def resolve_model(config):"]
        + [f"    value_{index} = config.get('key_{index}')" for index in range(50)]
        + ["    return value_1"]
    )
    payload = {
        "task_id": "replay",
        "events": [
            {
                "type": "observation",
                "observation_id": "source-1",
                "path": "model.py",
                "visible_content": source,
            },
            {"type": "prompt"},
            {"type": "action", "command": "rg -n resolve_model model.py"},
            {
                "type": "observation",
                "observation_id": "search-1",
                "causing_action": "rg resolve_model model.py",
                "visible_content": "model.py:1:def resolve_model(config):",
            },
            {"type": "prompt"},
        ],
    }
    config = AgentContextConfig.from_mapping(
        {
            "hot_observations": 1,
            "planner": {"mode": "retention", "target_retention": 0.6},
        }
    )

    result = replay_trace(payload, config)

    assert len(result.manifests) == 2
    assert result.report["model_forward_count"] == 0
    assert result.report["llm_token_count"] == 0
    assert result.canonical_messages[0]["content"] == source


def test_cli_validates_config_expands_matrix_and_replays_example(tmp_path: Path) -> None:
    config = REPO_ROOT / "configs" / "agent_context_typed_posterior.json"
    axes = REPO_ROOT / "configs" / "agent_context_ablation_axes.json"
    trace = REPO_ROOT / "examples" / "agent_context" / "demo_trace.json"
    validated = tmp_path / "validated.json"
    matrix = tmp_path / "matrix.json"
    replay = tmp_path / "replay.json"

    assert main(["validate-config", "--config", str(config), "--output", str(validated)]) == 0
    assert (
        main(
            [
                "matrix",
                "--config",
                str(config),
                "--axes",
                str(axes),
                "--output",
                str(matrix),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "replay",
                "--config",
                str(config),
                "--trace",
                str(trace),
                "--output",
                str(replay),
            ]
        )
        == 0
    )

    assert validated.is_file()
    assert matrix.read_text(encoding="utf-8").count('"name"') == 18
    replay_text = replay.read_text(encoding="utf-8")
    assert '"manifests"' in replay_text
    assert '"selected_level": "skeleton"' in replay_text


def test_core_modules_have_no_agent_or_model_sdk_imports() -> None:
    forbidden = {"minisweagent", "openai", "transformers", "torch", "litellm"}
    core_files = [
        path for path in (REPO_ROOT / "agent_context").rglob("*.py") if "adapters" not in path.parts
    ]
    imported: set[str] = set()
    for path in core_files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])

    assert forbidden.isdisjoint(imported)
