from __future__ import annotations

import argparse
import json
from pathlib import Path

from integrations.mini_swe_config import (
    DEFAULT_AGENT_STEP_LIMIT,
    adapt_config,
    load_yaml,
    write_yaml,
)
from integrations.mini_swe_config import (
    hosted_vllm_model_name as hosted_vllm_model_name,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a prompt-identical mini config for agent-context experiments"
    )
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8015/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--step-limit", type=int, default=DEFAULT_AGENT_STEP_LIMIT)
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
        step_limit=args.step_limit,
    )
    write_yaml(config, output)
    print(
        json.dumps(
            {
                "base_config": str(base_path),
                "output": str(output),
                "model": config["model"]["model_name"],
                "step_limit": config["agent"]["step_limit"],
                "legacy_pruner_removed": "pruner" not in config["agent"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
