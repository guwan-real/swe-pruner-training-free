from __future__ import annotations

from pathlib import Path

import pytest

from zero_forward_pruning.protocol import PruningRequest
from zero_forward_pruning.registry import METHODS, build_pruner
from zero_forward_pruning.store import RawStore


def _large_source() -> str:
    functions = []
    for index in range(90):
        name = "resolve_model" if index == 45 else f"unrelated_helper_{index}"
        body = [
            f"def {name}(config):",
            f"    marker_{index} = config.get('value_{index}')",
            "    if marker_%d is None:" % index,
            f"        raise ValueError('value_{index} is required')",
            f"    return marker_{index}",
        ]
        functions.append("\n".join(body))
    return "\n\n".join(functions)


def _request(*, threshold: float = 0.5, code: str | None = None) -> PruningRequest:
    return PruningRequest(
        query="How does resolve_model validate its model configuration?",
        code=code if code is not None else _large_source(),
        threshold=threshold,
        command="sed -n '1,900p' model_config.py",
        path="model_config.py",
        task="Fix missing validation in resolve_model.",
        request_id="method-test",
    )


@pytest.mark.parametrize("method", METHODS)
def test_every_method_is_reversible_and_zero_forward(tmp_path: Path, method: str) -> None:
    store = RawStore(tmp_path / method)
    pruner = build_pruner(
        method,
        raw_store=store,
        values={
            "min_input_tokens": 0,
            "min_savings_tokens": 1,
            "max_retention_ratio": 0.99,
            "max_cpu_ms": 1000,
            "public_base_url": f"http://host.docker.internal/{method}",
        },
    )
    request = _request()
    result = pruner.prune(request)
    assert result.status == "pruned", result.to_dict()
    assert result.model_forward_count == 0
    assert result.model_input_token_cnt == 0
    assert result.llm_token_count == 0
    assert result.left_token_cnt < result.origin_token_cnt
    assert "resolve_model" in result.pruned_code
    assert len(result.pruned_code) <= 9000
    assert result.raw_id is not None
    assert store.read(result.raw_id) == request.code
    assert "curl -fsS" in result.pruned_code


def test_adaptive_method_does_not_search_contract_threshold(tmp_path: Path) -> None:
    store = RawStore(tmp_path)
    pruner = build_pruner(
        "adaptive_evidence",
        raw_store=store,
        values={
            "min_input_tokens": 0,
            "min_savings_tokens": 1,
            "max_retention_ratio": 0.99,
            "max_cpu_ms": 1000,
        },
    )
    low = pruner.prune(_request(threshold=0.1))
    high = pruner.prune(_request(threshold=0.9))
    assert low.kept_line_numbers == high.kept_line_numbers
    assert low.diagnostics["contract_threshold_ignored"] is True
    assert high.diagnostics["contract_threshold_ignored"] is True


def test_output_cap_preserves_middle_intent_evidence(tmp_path: Path) -> None:
    noise = [
        (
            f"def unrelated_helper_{index}(value):\n"
            f"    first_{index} = value + {index}\n"
            f"    second_{index} = first_{index} * 2\n"
            f"    third_{index} = second_{index} - 1\n"
            f"    fourth_{index} = third_{index} or value\n"
            f"    return str(fourth_{index})"
        )
        for index in range(240)
    ]
    relevant = (
        "def resolve_model(config):\n"
        "    model_name = config['model_name']\n"
        "    if not model_name:\n"
        "        raise ValueError('model_name is required')\n"
        "    return model_name"
    )
    code = "\n\n".join([*noise[:120], relevant, *noise[120:]])
    pruner = build_pruner(
        "adaptive_evidence",
        raw_store=RawStore(tmp_path),
        values={
            "min_input_tokens": 0,
            "min_savings_tokens": 1,
            "max_retention_ratio": 0.99,
            "max_cpu_ms": 1000,
            "max_output_chars": 9000,
        },
    )
    result = pruner.prune(_request(code=code))
    assert result.status == "pruned", result.to_dict()
    assert result.diagnostics["output_char_cap_applied"] is True
    assert len(result.pruned_code) <= 9000
    assert "def resolve_model" in result.pruned_code
    assert "model_name is required" in result.pruned_code


def test_safe_rules_remains_query_independent_under_output_cap(tmp_path: Path) -> None:
    code = _large_source() * 4
    pruner = build_pruner(
        "safe_rules",
        raw_store=RawStore(tmp_path),
        values={
            "min_input_tokens": 0,
            "min_savings_tokens": 1,
            "max_retention_ratio": 0.99,
            "max_cpu_ms": 1000,
            "max_output_chars": 3000,
        },
    )
    first = pruner.prune(_request(code=code))
    second_request = PruningRequest(
        query="Find an entirely different symbol",
        code=code,
        threshold=0.5,
        command="cat unrelated.py",
        path="unrelated.py",
        task="Investigate something unrelated.",
    )
    second = pruner.prune(second_request)
    assert first.status == second.status == "pruned"
    assert first.kept_line_numbers == second.kept_line_numbers


def test_common_intent_term_does_not_expand_every_source_block(tmp_path: Path) -> None:
    code = _large_source() * 3
    request = PruningRequest(
        query="Inspect resolve_model config validation",
        code=code,
        command="sed -n '1,3000p' model_config.py",
        path="model_config.py",
        task="Fix resolve_model config validation.",
    )
    pruner = build_pruner(
        "adaptive_evidence",
        raw_store=RawStore(tmp_path),
        values={
            "min_input_tokens": 0,
            "min_savings_tokens": 1,
            "max_retention_ratio": 0.99,
            "max_cpu_ms": 1000,
        },
    )
    result = pruner.prune(request)
    assert result.status == "pruned", result.to_dict()
    assert "resolve_model" in result.pruned_code
    assert result.diagnostics["discriminative_intent_terms"] >= 1


def test_diff_and_short_output_fail_open(tmp_path: Path) -> None:
    pruner = build_pruner(
        "adaptive_evidence",
        raw_store=RawStore(tmp_path),
        values={"min_input_tokens": 100},
    )
    short = pruner.prune(_request(code="small output"))
    assert short.status == "skipped"
    assert short.pruned_code == "small output"
    diff = "\n".join(
        ["diff --git a/a.py b/a.py", "--- a/a.py", "+++ b/a.py", "@@ -1 +1 @@", "-a", "+b"] * 40
    )
    diff_result = pruner.prune(_request(code=diff))
    assert diff_result.status == "skipped"
    assert diff_result.pruned_code == diff
    assert diff_result.diagnostics["reason"] == "diff-is-never-pruned"


def test_recovery_store_failure_fails_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = RawStore(tmp_path)
    pruner = build_pruner(
        "adaptive_evidence",
        raw_store=store,
        values={
            "min_input_tokens": 0,
            "min_savings_tokens": 1,
            "max_retention_ratio": 0.99,
            "max_cpu_ms": 1000,
        },
    )

    def fail_save(*args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(store, "save", fail_save)
    result = pruner.prune(_request())
    assert result.status == "error"
    assert result.pruned_code == _request().code
    assert result.diagnostics["reason"] == "fail-open"
    assert result.model_forward_count == 0
