from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


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
) -> dict[str, Any]:
    """Generate one prompt-identical config for baseline and posterior arms."""

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
    # The official fork retains the focus-question prompt when its pruner
    # section is empty.  Remove the legacy client for both arms so prompt
    # parity is independent of the new history adapter.
    agent.pop("pruner", None)
    return config


def write_yaml(config: Mapping[str, Any], output: Path) -> None:
    import yaml

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(dict(config), allow_unicode=True, sort_keys=False, width=1000),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a prompt-identical mini config for posterior history experiments"
    )
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8015/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_path = Path(args.base_config).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    config = adapt_config(
        load_yaml(base_path),
        model_id=args.model_id,
        api_base=args.api_base,
        api_key=args.api_key,
        timeout=args.timeout,
    )
    write_yaml(config, output)
    print(
        json.dumps(
            {
                "base_config": str(base_path),
                "output": str(output),
                "model": config["model"]["model_name"],
                "legacy_pruner_removed": "pruner" not in config["agent"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
