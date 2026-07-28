from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from evaluation.replay import run_replay

from .budgets import LengthAwareBudget
from .io import load_requests, write_jsonl
from .protocol import BudgetConfig, PruningRequest
from .registry import available_methods, build_pruner


def _read_json(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _request_from_args(args: argparse.Namespace) -> PruningRequest:
    if args.request:
        return PruningRequest.from_dict(_read_json(args.request))
    if not args.text_file:
        raise ValueError("provide --request or --text-file")
    text = Path(args.text_file).read_text(encoding="utf-8")
    return PruningRequest(
        text=text,
        query=args.query or "",
        tool_type=args.tool_type,
        path=args.source_path,
        budget=BudgetConfig(
            keep_ratio=args.keep_ratio,
            no_prune_below=args.no_prune_below,
            context_window=args.context_window,
        ),
    )


def _add_method_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", required=True, choices=available_methods())
    parser.add_argument(
        "--config",
        help="method-specific JSON config (model paths remain local-only)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tf-prune",
        description="Training-free coding-agent observation pruning",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("methods", help="list the six implementations")

    prune_parser = subparsers.add_parser("prune", help="prune one observation")
    _add_method_args(prune_parser)
    source_group = prune_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--request", help="request JSON")
    source_group.add_argument("--text-file", help="raw observation text")
    prune_parser.add_argument("--query")
    prune_parser.add_argument("--tool-type", default="auto")
    prune_parser.add_argument("--source-path")
    prune_parser.add_argument("--keep-ratio", type=float, default=0.5)
    prune_parser.add_argument("--no-prune-below", type=int, default=20)
    prune_parser.add_argument("--context-window", type=int, default=1)
    prune_parser.add_argument("--output", help="result JSON; stdout by default")

    batch_parser = subparsers.add_parser("batch", help="prune request JSONL")
    _add_method_args(batch_parser)
    batch_parser.add_argument("--input", required=True)
    batch_parser.add_argument("--output", required=True)

    eval_parser = subparsers.add_parser(
        "evaluate",
        help="run labeled offline replay and emit metrics",
    )
    _add_method_args(eval_parser)
    eval_parser.add_argument("--input", required=True)
    eval_parser.add_argument("--output-dir", required=True)
    eval_parser.add_argument("--budget-schedule")
    eval_parser.add_argument(
        "--keep-ratio",
        type=float,
        help="override every replay row with one fixed keep ratio",
    )
    eval_parser.add_argument("--min-lines", type=int, default=1)
    eval_parser.add_argument("--no-prune-below", type=int, default=20)
    eval_parser.add_argument("--context-window", type=int, default=1)
    eval_parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "methods":
        for method in available_methods():
            print(method)
        return 0

    config = _read_json(args.config)
    pruner = build_pruner(args.method, config)
    if args.command == "prune":
        request = _request_from_args(args)
        result = pruner.prune(request).to_dict()
        serialized = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            Path(args.output).write_text(serialized, encoding="utf-8")
        else:
            sys.stdout.write(serialized)
        return 0

    if args.command == "batch":
        results = [pruner.prune(request) for request in load_requests(args.input)]
        write_jsonl(args.output, results)
        return 0

    if args.command == "evaluate":
        if args.budget_schedule and args.keep_ratio is not None:
            raise ValueError("--budget-schedule and --keep-ratio are mutually exclusive")
        schedule = (
            LengthAwareBudget.from_dict(_read_json(args.budget_schedule))
            if args.budget_schedule
            else (
                LengthAwareBudget.from_dict(
                    {
                        "bands": [
                            {
                                "max_lines": None,
                                "keep_ratio": args.keep_ratio,
                            }
                        ],
                        "min_lines": args.min_lines,
                        "no_prune_below": args.no_prune_below,
                        "context_window": args.context_window,
                    }
                )
                if args.keep_ratio is not None
                else None
            )
        )
        summary = run_replay(
            pruner,
            args.input,
            args.output_dir,
            budget_schedule=schedule,
            continue_on_error=args.continue_on_error,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
