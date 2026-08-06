from __future__ import annotations

import sys

import pytest

from agent_context.adapters.swebench import _guard_legacy_pruner_options


def test_framework_runner_rejects_prompt_changing_or_legacy_pruner_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["runner", "--disable-pruner"])
    with pytest.raises(SystemExit, match="breaks arm parity"):
        _guard_legacy_pruner_options()

    monkeypatch.setattr(sys, "argv", ["runner", "--pruner-url=http://legacy"])
    with pytest.raises(SystemExit, match="legacy SWE-Pruner"):
        _guard_legacy_pruner_options()


def test_framework_runner_allows_normal_swebench_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["runner", "--slice", "0:5", "--workers", "1"])

    _guard_legacy_pruner_options()
