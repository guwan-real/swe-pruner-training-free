from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from .protocol import PruningRequest, PruningResult


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            yield payload


def write_jsonl(
    path: str | Path,
    rows: Iterable[Mapping[str, Any] | PruningResult],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = row.to_dict() if isinstance(row, PruningResult) else dict(row)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_requests(path: str | Path) -> Iterator[PruningRequest]:
    for payload in read_jsonl(path):
        request_payload = payload.get("request", payload)
        if not isinstance(request_payload, Mapping):
            raise ValueError("request must be a JSON object")
        yield PruningRequest.from_dict(request_payload)
