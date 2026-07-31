from __future__ import annotations

import argparse
import json
import tempfile
import threading
import urllib.request
from pathlib import Path

from zero_forward_pruning.http_server import ZeroForwardHTTPServer
from zero_forward_pruning.registry import build_pruner
from zero_forward_pruning.store import RawStore


def _json_request(url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError(f"{url} did not return a JSON object")
    return value


def probe_service(base_url: str) -> dict:
    base = base_url.rstrip("/")
    health = _json_request(f"{base}/health")
    if health.get("status") != "healthy":
        raise RuntimeError(f"service is not healthy: {health}")
    if health.get("model_forward_count") != 0 or health.get("llm_token_count") != 0:
        raise RuntimeError(f"health response violates zero-forward invariant: {health}")
    relevant = [
        "def resolve_model(config):",
        "    model_name = config['model_name']",
        "    if not model_name:",
        "        raise ValueError('model_name is required')",
        "    return model_name",
    ]
    noise = [
        (
            f"def unrelated_helper_{index}(value):\n"
            f"    first_{index} = value + {index}\n"
            f"    second_{index} = first_{index} * 2\n"
            f"    third_{index} = second_{index} - 1\n"
            f"    fourth_{index} = third_{index} or value\n"
            f"    fifth_{index} = str(fourth_{index})\n"
            f"    return fifth_{index}"
        )
        for index in range(220)
    ]
    raw = "\n\n".join([*noise[:110], "\n".join(relevant), *noise[110:]])
    result = _json_request(
        f"{base}/prune",
        {
            "query": "How does resolve_model validate model_name?",
            "code": raw,
            "threshold": 0.5,
            "command": "sed -n '1,999p' model_config.py",
            "path": "model_config.py",
            "task": "Fix model configuration validation.",
            "request_id": "zero-forward-preflight",
            "metadata": {"traffic_class": "preflight"},
        },
    )
    for field in ("model_input_token_cnt", "model_forward_count", "llm_token_count"):
        if result.get(field) != 0:
            raise RuntimeError(f"{field} must be zero: {result}")
    if result.get("status") != "pruned":
        raise RuntimeError(f"preflight fixture was not pruned: {result}")
    if health.get("method") != "safe_rules" and "resolve_model" not in str(
        result.get("pruned_code", "")
    ):
        raise RuntimeError("task-relevant function was not retained")
    raw_id = result.get("raw_id")
    if not isinstance(raw_id, str):
        raise RuntimeError("pruned result has no raw_id")
    with urllib.request.urlopen(f"{base}/raw/{raw_id}", timeout=10) as response:
        recovered = response.read().decode("utf-8")
    if recovered != raw:
        raise RuntimeError("raw recovery was not byte-for-byte equivalent")
    return {
        "status": "passed",
        "method": result.get("method"),
        "origin_token_cnt": result.get("origin_token_cnt"),
        "left_token_cnt": result.get("left_token_cnt"),
        "model_forward_count": 0,
        "llm_token_count": 0,
        "recovery_verified": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe a running zero-forward pruning service")
    parser.add_argument("--url", default="http://127.0.0.1:8121")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Start an ephemeral local service and verify pruning plus byte-exact recovery",
    )
    return parser


def self_test() -> dict:
    with tempfile.TemporaryDirectory(prefix="zero-forward-preflight-") as directory:
        store = RawStore(Path(directory))
        pruner = build_pruner(
            "adaptive_evidence",
            raw_store=store,
            values={
                "public_base_url": "http://host.docker.internal:1",
                "max_cpu_ms": 500.0,
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
        try:
            host, port = server.server_address
            return probe_service(f"http://{host}:{port}")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = self_test() if args.self_test else probe_service(args.url)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
