from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tf_pruning.io import load_requests, write_jsonl

from . import build_pruner


def _load_config(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("config must be a JSON object")
    nested = payload.get("conditional_ppl")
    if isinstance(nested, dict):
        return nested
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Coarse-to-fine conditional-surprisal pruning",
    )
    parser.add_argument("--config", required=True, help="JSON configuration")
    parser.add_argument("--input", required=True, help="Input request JSONL")
    parser.add_argument("--output", required=True, help="Output result JSONL")
    args = parser.parse_args(argv)

    pruner = build_pruner(_load_config(args.config))
    write_jsonl(args.output, (pruner.prune(row) for row in load_requests(args.input)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
