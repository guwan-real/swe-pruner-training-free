from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from posterior_pruning.http_server import PosteriorPruningHTTPServer
from posterior_pruning.methods.common import unchanged_result


class EchoMethod:
    name = "echo"

    def prune(self, request):
        return unchanged_result(
            self.name,
            request,
            status="skipped",
            started_at=time.perf_counter(),
        )


class FailingMethod:
    name = "failing"

    def prune(self, request):
        raise RuntimeError("scorer unavailable")


def payload() -> dict:
    return {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "first action"},
            {"role": "user", "content": "Observation:\nfull response"},
        ],
        "observation_index": 3,
        "next_action": "second action",
        "keep_ratio": 0.5,
    }


def request_json(url: str, value: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(value).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=2) as response:
        return response.status, json.load(response)


@pytest.fixture
def server_factory():
    servers = []

    def factory(method, *, fail_open=True):
        server = PosteriorPruningHTTPServer(
            ("127.0.0.1", 0),
            method=method,
            model_id="test-model",
            min_chars=0,
            fail_open=fail_open,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append((server, thread))
        return f"http://127.0.0.1:{server.server_port}"

    yield factory

    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_post_action_endpoint_is_distinct_from_legacy_prune(server_factory) -> None:
    base = server_factory(EchoMethod())

    status, result = request_json(f"{base}/prune-post-action", payload())

    assert status == 200
    assert result["status"] == "skipped"
    assert result["pruned_response"] == "Observation:\nfull response"
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        request_json(f"{base}/prune", payload())
    assert exc_info.value.code == 409


def test_http_service_fails_open_with_original_observation(server_factory) -> None:
    base = server_factory(FailingMethod(), fail_open=True)

    status, result = request_json(f"{base}/prune-post-action", payload())

    assert status == 200
    assert result["status"] == "error"
    assert result["pruned_response"] == "Observation:\nfull response"
    assert "scorer unavailable" in result["error"]
