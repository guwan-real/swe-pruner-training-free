from __future__ import annotations

from evaluation.convert_existing import convert_row


def test_structured_row_uses_relative_label_indices() -> None:
    converted = convert_row(
        {
            "sample_id": "s1",
            "code": "line a\nline b\nline c",
            "query": "b",
            "file_path": "pkg/mod.py",
            "line_numbers": [100, 101, 102],
            "line_keep_labels": [0, 1, 1],
            "line_confidences": [0.7, 0.95, 0.8],
        },
        row_index=1,
        required_confidence=0.9,
        keep_ratio=0.5,
        no_prune_below=0,
    )
    assert converted["gold_line_numbers"] == [2, 3]
    assert converted["required_line_numbers"] == [2]
    assert converted["request"]["metadata"]["original_start_line"] == 100


def test_official_fragments_accept_line_numbers() -> None:
    converted = convert_row(
        {
            "code": "a\nb\nc",
            "query": "c",
            "kept_frags": ["2", "c"],
        },
        row_index=2,
        required_confidence=0.9,
        keep_ratio=0.5,
        no_prune_below=0,
    )
    assert converted["gold_line_numbers"] == [2, 3]
    assert converted["required_line_numbers"] == []
