from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

MATRIX_FIELDS = (
    "budget",
    "method",
    "samples",
    "retention_ratio",
    "estimated_token_retention_ratio",
    "macro_line_recall",
    "macro_line_f1",
    "macro_required_line_recall",
    "critical_miss_rate",
    "latency_mean_ms",
    "latency_p95_ms",
    "model_forward_count",
)


def load_matrix(root: str | Path) -> list[dict[str, Any]]:
    matrix_root = Path(root)
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(matrix_root.glob("keep_*/summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        latency = payload.get("latency_ms", {})
        rows.append(
            {
                "budget": summary_path.parent.name.removeprefix("keep_"),
                "method": payload.get("method"),
                "samples": payload.get("samples"),
                "retention_ratio": payload.get("retention_ratio"),
                "estimated_token_retention_ratio": payload.get("estimated_token_retention_ratio"),
                "macro_line_recall": payload.get("macro_line_recall"),
                "macro_line_f1": payload.get("macro_line_f1"),
                "macro_required_line_recall": payload.get("macro_required_line_recall"),
                "critical_miss_rate": payload.get("critical_miss_rate"),
                "latency_mean_ms": latency.get("mean"),
                "latency_p95_ms": latency.get("p95"),
                "model_forward_count": payload.get("model_forward_count"),
            }
        )
    return rows


def mark_recall_retention_pareto(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    marked: list[dict[str, Any]] = []
    for row in rows:
        retention = row.get("estimated_token_retention_ratio")
        recall = row.get("macro_required_line_recall")
        pareto = None
        if retention is not None and recall is not None:
            pareto = not any(
                other is not row
                and other.get("estimated_token_retention_ratio") is not None
                and other.get("macro_required_line_recall") is not None
                and float(other["estimated_token_retention_ratio"]) <= float(retention)
                and float(other["macro_required_line_recall"]) >= float(recall)
                and (
                    float(other["estimated_token_retention_ratio"]) < float(retention)
                    or float(other["macro_required_line_recall"]) > float(recall)
                )
                for other in rows
            )
        marked.append({**row, "pareto": pareto})
    return marked


def write_matrix(root: str | Path) -> list[dict[str, Any]]:
    matrix_root = Path(root)
    rows = mark_recall_retention_pareto(load_matrix(matrix_root))
    (matrix_root / "matrix.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (matrix_root / "matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(*MATRIX_FIELDS, "pareto"),
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect keep-ratio replay summaries into one Pareto table"
    )
    parser.add_argument("output_root")
    args = parser.parse_args(argv)
    rows = write_matrix(args.output_root)
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
