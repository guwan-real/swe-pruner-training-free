from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from zero_forward_pruning.http_server import ZeroForwardHTTPServer
from zero_forward_pruning.mini_adapter.client import (
    ZeroForwardClient,
    ZeroForwardClientConfig,
)
from zero_forward_pruning.registry import build_pruner
from zero_forward_pruning.store import RawStore


def _source() -> str:
    return "\n\n".join(
        [
            (
                f"def {'resolve_model' if index == 40 else f'helper_{index}'}(config):\n"
                f"    value = config.get('key_{index}')\n"
                f"    if value is None:\n        raise ValueError('key_{index}')\n"
                "    return value"
            )
            for index in range(80)
        ]
    )


@pytest.fixture
def service(tmp_path: Path):
    store = RawStore(tmp_path)
    pruner = build_pruner(
        "adaptive_evidence",
        raw_store=store,
        values={
            "min_input_tokens": 0,
            "min_savings_tokens": 1,
            "max_retention_ratio": 0.99,
            "max_cpu_ms": 1000,
            "public_base_url": "http://host.docker.internal:9999",
        },
    )
    server = ZeroForwardHTTPServer(
        ("127.0.0.1", 0),
        pruner=pruner,
        raw_store=store,
        fail_open=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}", store
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def test_http_contract_metrics_and_raw_recovery(service) -> None:
    base, store = service
    client = ZeroForwardClient(ZeroForwardClientConfig(url=base, timeout=5, min_chars=0))
    raw = _source()
    result = client.prune(
        code=raw,
        query="resolve_model",
        command="cat model.py",
        path="model.py",
        task="validate resolve_model",
    )
    assert result["status"] == "pruned"
    assert result["model_input_token_cnt"] == 0
    assert result["model_forward_count"] == 0
    assert result["llm_token_count"] == 0
    assert store.read(result["raw_id"]) == raw
    with urllib.request.urlopen(f"{base}/raw/{result['raw_id']}", timeout=5) as response:
        assert response.read().decode() == raw
    metrics = _get_json(f"{base}/metrics")
    assert metrics["requests"] == 1
    assert metrics["model_forward_count"] == 0
    assert metrics["llm_token_count"] == 0
    assert metrics["estimated_tokens_saved"] > 0


def test_removed_post_action_endpoint_is_not_available(service) -> None:
    base, _ = service
    request = urllib.request.Request(
        f"{base}/prune-post-action",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request, timeout=5)
    assert error.value.code == 404


def test_http_rejects_bad_contract(service) -> None:
    base, _ = service
    request = urllib.request.Request(
        f"{base}/prune",
        data=json.dumps({"query": 1, "code": "x"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request, timeout=5)
    assert error.value.code == 400
