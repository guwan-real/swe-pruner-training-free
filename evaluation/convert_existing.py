from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tf_pruning.io import read_jsonl, write_jsonl


def _positive_indices(labels: Sequence[Any]) -> list[int]:
    return [index for index, label in enumerate(labels, start=1) if bool(label)]


def _fragment_indices(code: str, fragments: Sequence[Any]) -> list[int]:
    lines = code.splitlines()
    selected: set[int] = set()
    for fragment in fragments:
        value = str(fragment)
        if value.isdigit():
            line_no = int(value)
            if 1 <= line_no <= len(lines):
                selected.add(line_no)
            continue
        normalized = value.strip()
        if not normalized:
            continue
        selected.update(
            index for index, line in enumerate(lines, start=1) if line.strip() == normalized
        )
    return sorted(selected)


def convert_row(
    row: Mapping[str, Any],
    *,
    row_index: int,
    required_confidence: float,
    keep_ratio: float,
    no_prune_below: int,
) -> dict[str, Any]:
    if "code" in row:
        code = str(row["code"])
    elif "document" in row:
        code = str(row["document"])
    else:
        raise ValueError(f"row {row_index}: missing code/document")

    if "line_keep_labels" in row:
        labels = list(row["line_keep_labels"])
        gold = _positive_indices(labels)
    elif "labels" in row:
        labels = list(row["labels"])
        gold = _positive_indices(labels)
    elif "kept_frags" in row:
        labels = []
        gold = _fragment_indices(code, list(row["kept_frags"]))
    else:
        raise ValueError(f"row {row_index}: missing line_keep_labels/labels/kept_frags")

    if labels and len(labels) != len(code.splitlines()):
        raise ValueError(
            f"row {row_index}: {len(labels)} labels for {len(code.splitlines())} lines"
        )

    confidences = list(row.get("line_confidences", ()))
    required = (
        [
            line_no
            for line_no in gold
            if line_no <= len(confidences)
            and float(confidences[line_no - 1]) >= required_confidence
        ]
        if confidences
        else []
    )
    if confidences and len(confidences) != len(code.splitlines()):
        raise ValueError(f"row {row_index}: confidence count does not match code lines")

    request_id = str(row.get("sample_id") or row.get("task_id") or f"converted-{row_index}")
    original_line_numbers = row.get("line_numbers", ())
    metadata: dict[str, Any] = {
        "dataset_source": row.get("dataset_source"),
        "repo_name": row.get("repo_name"),
        "task_id": row.get("task_id"),
        "anchor_symbol": row.get("anchor_symbol"),
    }
    if original_line_numbers:
        metadata["original_start_line"] = int(original_line_numbers[0])
        metadata["original_end_line"] = int(original_line_numbers[-1])
    metadata = {key: value for key, value in metadata.items() if value is not None}

    return {
        "request": {
            "request_id": request_id,
            "text": code,
            "query": str(row.get("query", "")),
            "tool_type": "source",
            "path": row.get("file_path"),
            "budget": {
                "keep_ratio": keep_ratio,
                "min_lines": 1,
                "no_prune_below": no_prune_below,
                "context_window": 1,
            },
            "metadata": metadata,
        },
        "gold_line_numbers": gold,
        "required_line_numbers": required,
    }


def convert_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    required_confidence: float = 0.9,
    keep_ratio: float = 0.5,
    no_prune_below: int = 20,
    limit: int | None = None,
) -> int:
    rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(read_jsonl(input_path), start=1):
        if limit is not None and len(rows) >= limit:
            break
        rows.append(
            convert_row(
                row,
                row_index=row_index,
                required_confidence=required_confidence,
                keep_ratio=keep_ratio,
                no_prune_below=no_prune_below,
            )
        )
    write_jsonl(output_path, rows)
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert existing SWE-Pruner training rows to replay JSONL"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--required-confidence", type=float, default=0.9)
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--no-prune-below", type=int, default=20)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    count = convert_file(
        args.input,
        args.output,
        required_confidence=args.required_confidence,
        keep_ratio=args.keep_ratio,
        no_prune_below=args.no_prune_below,
        limit=args.limit,
    )
    print(
        json.dumps(
            {"converted_rows": count, "output": args.output},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
