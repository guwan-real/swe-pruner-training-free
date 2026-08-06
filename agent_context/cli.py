from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agent_context.config import AgentContextConfig
from agent_context.experiments import expand_ablation_matrix
from agent_context.registry import DEFAULT_COMPONENT_REGISTRY
from agent_context.replay import replay_trace


def _read_object(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON file must contain an object: {resolved}")
    return dict(payload)


def _emit(payload: Any, output: str | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if output is None:
        print(text, end="")
        return
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model-independent coding-agent context runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("components", help="List registered experiment components")
    validate = subparsers.add_parser("validate-config", help="Validate and normalize a config")
    validate.add_argument("--config", required=True)
    validate.add_argument("--output")
    matrix = subparsers.add_parser("matrix", help="Expand orthogonal ablation axes")
    matrix.add_argument("--config", required=True)
    matrix.add_argument("--axes", required=True)
    matrix.add_argument("--prefix", default="agent-context")
    matrix.add_argument("--output")
    replay = subparsers.add_parser("replay", help="Replay a structured event trace")
    replay.add_argument("--config", required=True)
    replay.add_argument("--trace", required=True)
    replay.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "components":
        _emit(DEFAULT_COMPONENT_REGISTRY.manifest(), None)
        return 0
    config = AgentContextConfig.from_mapping(_read_object(args.config))
    if args.command == "validate-config":
        DEFAULT_COMPONENT_REGISTRY.components(config)
        _emit(config.to_dict(), args.output)
        return 0
    if args.command == "matrix":
        axes = _read_object(args.axes)
        arms = expand_ablation_matrix(config, axes, prefix=args.prefix)
        _emit([arm.to_dict() for arm in arms], args.output)
        return 0
    result = replay_trace(_read_object(args.trace), config)
    _emit(result.to_dict(), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
