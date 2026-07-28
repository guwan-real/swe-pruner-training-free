from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tf_pruning.protocol import PruningRequest

from . import build_pruner


def _load_config(path: str | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("config must contain a JSON object")
    return payload


def _input_lines(path: str) -> Iterable[str]:
    if path == "-":
        yield from sys.stdin
        return
    with Path(path).open("r", encoding="utf-8") as handle:
        yield from handle


def _output_handle(path: str):
    if path == "-":
        return sys.stdout
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target.open("w", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Training-free BM25 + structural-anchor line pruning",
    )
    parser.add_argument("--input", required=True, help="request JSONL path or -")
    parser.add_argument("--output", default="-", help="result JSONL path or -")
    parser.add_argument("--config", help="optional JSON configuration")
    args = parser.parse_args(argv)

    pruner = build_pruner(_load_config(args.config))
    output = _output_handle(args.output)
    should_close = output is not sys.stdout
    try:
        for source_line_no, raw_line in enumerate(_input_lines(args.input), start=1):
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{args.input}:{source_line_no}: expected a JSON object")
            request_payload = payload.get("request", payload)
            if not isinstance(request_payload, Mapping):
                raise ValueError(f"{args.input}:{source_line_no}: request must be an object")
            result = pruner.prune(PruningRequest.from_dict(request_payload))
            output.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
    finally:
        if should_close:
            output.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
