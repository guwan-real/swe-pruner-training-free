from __future__ import annotations

import pytest

from posterior_pruning.candidates import (
    CandidateConfig,
    candidate_for_ratio,
    render_kept_lines,
)
from posterior_pruning.protocol import PosteriorPruningRequest


def payload() -> dict:
    return {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "first action"},
            {"role": "user", "content": "Observation:\none\ntwo"},
        ],
        "observation_index": 3,
        "next_action": "```bash\nsed -n '1,2p' file.py\n```",
        "keep_ratio": 0.5,
    }


def test_post_action_request_replaces_only_the_observation() -> None:
    request = PosteriorPruningRequest.from_dict(payload())

    messages = request.messages_with_observation("Observation:\none")

    assert request.observation == "Observation:\none\ntwo"
    assert messages[3]["content"] == "Observation:\none"
    assert messages[0]["content"] == "system"
    assert len(messages) == 4


def test_post_action_request_requires_a_user_observation() -> None:
    value = payload()
    value["observation_index"] = 2

    with pytest.raises(ValueError, match="role='user'"):
        PosteriorPruningRequest.from_dict(value)


def test_candidate_proposal_is_self_contained_and_preserves_line_order() -> None:
    text = "\n".join(("head", "noise", "ERROR parser.py:7 failed", "tail"))
    candidate = candidate_for_ratio(
        text,
        next_action="open parser.py",
        keep_ratio=0.25,
        config=CandidateConfig(
            block_max_lines=1,
            protect_errors=True,
            protect_diffs=False,
            protect_edge_lines=False,
        ),
    )

    assert 3 in candidate.kept_line_numbers
    assert "ERROR parser.py:7 failed" in candidate.text
    assert candidate.kept_line_numbers == tuple(sorted(candidate.kept_line_numbers))
    assert render_kept_lines(text.splitlines(), (1, 4)).splitlines() == [
        "head",
        "... [posterior-pruned lines 2-3] ...",
        "tail",
    ]


def test_edge_protection_does_not_protect_an_entire_short_observation() -> None:
    text = "\n".join(f"line {index} with repeated content" for index in range(1, 9))

    candidate = candidate_for_ratio(
        text,
        next_action="unrelated action",
        keep_ratio=0.5,
        config=CandidateConfig(block_max_lines=12, protect_edge_lines=True),
    )

    assert 1 in candidate.kept_line_numbers
    assert 8 in candidate.kept_line_numbers
    assert len(candidate.kept_line_numbers) < 8
