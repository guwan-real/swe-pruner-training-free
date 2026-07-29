from __future__ import annotations

import argparse
import copy
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def discover_base_config(explicit: str | None = None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
    elif os.getenv("MINI_SWE_BASE_CONFIG"):
        path = Path(os.environ["MINI_SWE_BASE_CONFIG"]).expanduser().resolve()
    else:
        try:
            from minisweagent.config import builtin_config_dir
        except ImportError as exc:
            raise RuntimeError(
                "mini-swe-agent is not importable in the active conda environment"
            ) from exc
        path = (Path(builtin_config_dir) / "extra" / "swebench.yaml").resolve()
    if not path.is_file():
        raise FileNotFoundError(f"mini-swe-agent base config does not exist: {path}")
    return path


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required; install the project with python -m pip install -e '.[agent]'"
        ) from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"mini-swe-agent config must be a YAML object: {path}")
    return payload


def validate_pruning_contract(config: Mapping[str, Any]) -> None:
    agent = config.get("agent")
    if not isinstance(agent, Mapping):
        raise ValueError("base config has no agent section")
    pruner = agent.get("pruner")
    if not isinstance(pruner, Mapping):
        raise ValueError(
            "installed mini-swe-agent config has no agent.pruner hook; "
            "use a pruning-capable mini-swe-agent config"
        )
    prompt_text = "\n".join(
        str(agent.get(key, ""))
        for key in ("system_template", "instance_template", "format_error_template")
    )
    if "context_focus_question" not in prompt_text:
        raise ValueError(
            "base config does not ask the agent for context_focus_question; "
            "the pruner would never receive a query"
        )


def resolve_pruning_base_config(
    primary: Path | None,
    *,
    search_root: Path | None = None,
    explicit: bool = False,
) -> Path:
    """Resolve one compatible config without guessing between multiple templates."""

    primary_error: Exception | None = None
    if primary is not None:
        primary = primary.expanduser().resolve()
        try:
            if not primary.is_file():
                raise FileNotFoundError(f"base config does not exist: {primary}")
            validate_pruning_contract(load_yaml(primary))
            return primary
        except (OSError, RuntimeError, ValueError) as exc:
            if explicit:
                raise
            primary_error = exc

    compatible: list[Path] = []
    if search_root is not None:
        search_root = search_root.expanduser().resolve()
        search_dirs = (
            search_root / "templates",
            search_root / "src" / "minisweagent" / "config" / "extra",
        )
        for directory in search_dirs:
            if not directory.is_dir():
                continue
            for candidate in sorted(directory.rglob("*.yaml")):
                if primary is not None and candidate.resolve() == primary:
                    continue
                try:
                    validate_pruning_contract(load_yaml(candidate))
                except (OSError, RuntimeError, ValueError):
                    continue
                compatible.append(candidate.resolve())

    compatible = sorted(set(compatible))
    if len(compatible) == 1:
        return compatible[0]
    if len(compatible) > 1:
        rendered = "\n".join(f"  - {path}" for path in compatible[:20])
        raise RuntimeError(
            "multiple pruning-capable mini-swe-agent configs were found; "
            "set MINI_SWE_BASE_CONFIG explicitly:\n"
            f"{rendered}"
        )
    if primary_error is not None:
        raise RuntimeError(
            "the installed mini-swe-agent base config is not pruning-capable and "
            "no unique compatible template was found; set MINI_SWE_BASE_CONFIG. "
            f"Primary config error: {primary_error}"
        ) from primary_error
    raise RuntimeError(
        "no pruning-capable mini-swe-agent config was found; "
        "set MINI_SWE_BASE_CONFIG to the existing pruning template"
    )


def hosted_vllm_model_name(model_id: str) -> str:
    model_id = model_id.strip()
    if not model_id:
        raise ValueError("vLLM model id must not be empty")
    if model_id.startswith(("hosted_vllm/", "openai/")):
        return model_id
    return f"hosted_vllm/{model_id}"


def adapt_config(
    base: Mapping[str, Any],
    *,
    model_id: str,
    api_base: str,
    pruner_url: str,
    keep_ratio: float,
    api_key: str = "EMPTY",
    min_chars: int = 500,
    timeout: float = 180.0,
) -> dict[str, Any]:
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be in (0, 1]")
    if min_chars < 0:
        raise ValueError("min_chars must be non-negative")
    validate_pruning_contract(base)
    config = copy.deepcopy(dict(base))

    model = config.setdefault("model", {})
    if not isinstance(model, dict):
        raise ValueError("base config model section must be an object")
    model["model_name"] = hosted_vllm_model_name(model_id)
    model["cost_tracking"] = "ignore_errors"
    model.pop("set_cache_control", None)
    kwargs = model.setdefault("model_kwargs", {})
    if not isinstance(kwargs, dict):
        raise ValueError("base config model.model_kwargs must be an object")
    kwargs.update(
        {
            "api_base": api_base.rstrip("/"),
            "api_key": (api_key if api_key == "EMPTY" else "${MSWEA_MODEL_API_KEY}"),
            "drop_params": True,
            "temperature": 0.0,
            "timeout": timeout,
        }
    )

    agent = config["agent"]
    pruner = agent["pruner"]
    pruner.update(
        {
            "url": pruner_url,
            "threshold": 1.0 - keep_ratio,
            "timeout": timeout,
            "retries": 3,
            "min_chars": min_chars,
        }
    )
    return config


def write_yaml(config: Mapping[str, Any], output: Path) -> None:
    import yaml

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(
            dict(config),
            allow_unicode=True,
            sort_keys=False,
            width=1000,
        ),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Patch the installed pruning-capable mini-swe-agent config for local vLLM"
    )
    parser.add_argument("--base-config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8015/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--pruner-url", required=True)
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--min-chars", type=int, default=500)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_path = discover_base_config(args.base_config)
    base = load_yaml(base_path)
    adapted = adapt_config(
        base,
        model_id=args.model_id,
        api_base=args.api_base,
        api_key=args.api_key,
        pruner_url=args.pruner_url,
        keep_ratio=args.keep_ratio,
        min_chars=args.min_chars,
        timeout=args.timeout,
    )
    output = Path(args.output).resolve()
    write_yaml(adapted, output)
    print(
        json.dumps(
            {
                "base_config": str(base_path),
                "output": str(output),
                "model_name": adapted["model"]["model_name"],
                "api_base": adapted["model"]["model_kwargs"]["api_base"],
                "pruner_url": adapted["agent"]["pruner"]["url"],
                "keep_ratio": args.keep_ratio,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
