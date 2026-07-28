from __future__ import annotations

from scripts.extract_hf_signals import (
    build_prompt,
    response_token_lines,
    token_indices_in_span,
)


def test_prompt_spans_and_response_line_mapping() -> None:
    request = {
        "query": "fix timeout",
        "tool_type": "source",
        "path": "client.py",
        "text": "alpha\nbeta",
    }
    layout = build_prompt(request)
    assert layout.prompt[layout.query_start : layout.query_end] == "fix timeout"
    assert layout.prompt[layout.response_start : layout.response_end] == ("alpha\nbeta")

    offsets = [
        (0, 0),
        (layout.query_start, layout.query_end),
        (layout.response_start, layout.response_start + 5),
        (layout.response_start + 6, layout.response_end),
    ]
    indices, lines = response_token_lines(
        offsets,
        layout=layout,
        response=request["text"],
    )
    assert indices == [2, 3]
    assert lines == [1, 2]
    assert token_indices_in_span(
        offsets,
        layout.query_start,
        layout.query_end,
    ) == [1]
