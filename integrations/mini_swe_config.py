from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

DEFAULT_AGENT_STEP_LIMIT = 100


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required: pip install -e '.[agent]'") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"mini-swe-agent config must be a YAML object: {path}")
    return payload


def hosted_vllm_model_name(model_id: str) -> str:
    model_id = model_id.strip()
    if not model_id:
        raise ValueError("vLLM model id must not be empty")
    return (
        model_id if model_id.startswith(("hosted_vllm/", "openai/")) else f"hosted_vllm/{model_id}"
    )


def adapt_config(
    base: Mapping[str, Any],
    *,
    model_id: str,
    api_base: str,
    api_key: str = "EMPTY",
    timeout: float = 180.0,
    step_limit: int = DEFAULT_AGENT_STEP_LIMIT,
) -> dict[str, Any]:
    """Create one prompt-identical mini-swe config for experiment arms."""

    if step_limit <= 0:
        raise ValueError("step_limit must be positive")

    config = copy.deepcopy(dict(base))
    model = config.setdefault("model", {})
    agent = config.setdefault("agent", {})
    if not isinstance(model, dict) or not isinstance(agent, dict):
        raise ValueError("base config model and agent sections must be objects")
    model["model_name"] = hosted_vllm_model_name(model_id)
    model["cost_tracking"] = "ignore_errors"
    model.pop("set_cache_control", None)
    kwargs = model.setdefault("model_kwargs", {})
    if not isinstance(kwargs, dict):
        raise ValueError("base config model.model_kwargs must be an object")
    kwargs.update(
        {
            "api_base": api_base.rstrip("/"),
            "api_key": api_key,
            "drop_params": True,
            "temperature": 0.0,
            "timeout": timeout,
        }
    )
    # Removing only the legacy client preserves the fork's context-focus prompt.
    agent.pop("pruner", None)
    agent["step_limit"] = step_limit
    return config


def write_yaml(config: Mapping[str, Any], output: Path) -> None:
    import yaml

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(dict(config), allow_unicode=True, sort_keys=False, width=1000),
        encoding="utf-8",
    )
