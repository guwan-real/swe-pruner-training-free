from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_posterior_launcher_is_isolated_and_has_five_default_arms() -> None:
    text = (ROOT / "scripts" / "run_posterior_swebench.sh").read_text()

    assert "scripts/run_server_experiments.sh" in text
    assert "single_verify,budget_search,greedy_blocks,block_influence" in text
    assert "POSTERIOR_PRUNER_URL" in text
    assert "posterior_pruning.mini_adapter.swebench" in text
    assert "disable_uv_or_venv" in text
    assert "result)" in text


def test_posterior_launcher_does_not_use_legacy_prune_endpoint() -> None:
    text = (ROOT / "scripts" / "run_posterior_swebench.sh").read_text()

    assert "--pruner-url" not in text
    assert "/prune-post-action" not in text
