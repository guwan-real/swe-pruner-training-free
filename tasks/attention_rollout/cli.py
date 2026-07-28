from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tf_pruning.io import load_requests
from tf_pruning.protocol import PruningResult

from . import build_pruner


def _load_config(path: str | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("config must be a JSON object")
    nested = payload.get("pruner", payload)
    if not isinstance(nested, Mapping):
        raise ValueError("config.pruner must be a JSON object")
    return nested


def _write_results(
    rows: Iterable[PruningResult],
    output: str,
) -> None:
    if output == "-":
        for row in rows:
            sys.stdout.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
        return
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prune JSONL observations with attention mass or rollout."
    )
    parser.add_argument("input", help="Input JSONL of PruningRequest objects")
    parser.add_argument(
        "-o",
        "--output",
        default="-",
        help="Output JSONL path; '-' writes to stdout",
    )
    parser.add_argument("--config", help="Path to config JSON")
    args = parser.parse_args(argv)

    pruner = build_pruner(_load_config(args.config))
    _write_results(
        (pruner.prune(request) for request in load_requests(args.input)),
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
