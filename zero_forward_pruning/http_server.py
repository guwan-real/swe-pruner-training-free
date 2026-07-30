from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from zero_forward_pruning.methods.common import unchanged_result
from zero_forward_pruning.protocol import PruningRequest, PruningResult
from zero_forward_pruning.registry import METHODS, build_pruner
from zero_forward_pruning.store import RawStore

LOGGER = logging.getLogger("zero_forward_pruning.http_server")
MAX_BODY_BYTES = 64 * 1024 * 1024


@dataclass
class ServiceMetrics:
    requests: int = 0
    pruned: int = 0
    skipped: int = 0
    errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cpu_latency_ms: float = 0.0
    model_forward_count: int = 0
    llm_token_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, result: PruningResult) -> None:
        with self._lock:
            self.requests += 1
            if result.status == "pruned":
                self.pruned += 1
            elif result.status == "error":
                self.errors += 1
            else:
                self.skipped += 1
            self.input_tokens += result.origin_token_cnt
            self.output_tokens += result.left_token_cnt
            self.cpu_latency_ms += result.latency_ms
            self.model_forward_count += result.model_forward_count
            self.llm_token_count += result.llm_token_count

    def to_dict(self) -> dict[str, int | float]:
        with self._lock:
            return {
                "requests": self.requests,
                "pruned": self.pruned,
                "skipped": self.skipped,
                "errors": self.errors,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "estimated_tokens_saved": self.input_tokens - self.output_tokens,
                "cpu_latency_ms": self.cpu_latency_ms,
                "model_forward_count": self.model_forward_count,
                "llm_token_count": self.llm_token_count,
            }


class ZeroForwardHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        pruner: Any,
        raw_store: RawStore,
        fail_open: bool,
    ) -> None:
        super().__init__(server_address, ZeroForwardHandler)
        self.pruner = pruner
        self.raw_store = raw_store
        self.fail_open = fail_open
        self.metrics = ServiceMetrics()


class ZeroForwardHandler(BaseHTTPRequestHandler):
    server: ZeroForwardHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), format % args)

    def _send_json(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, status: HTTPStatus, text: str) -> None:
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _read_payload(self) -> Mapping[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("Content-Length must be an integer") from exc
        if length < 0 or length > MAX_BODY_BYTES:
            raise ValueError(f"request body must be between 0 and {MAX_BODY_BYTES} bytes")
        try:
            payload = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"request body must be valid JSON: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("request body must be a JSON object")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path.rstrip("/")
        if path == "/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "healthy",
                    "service": "zero-forward-pruning",
                    "method": self.server.pruner.name,
                    "contract": "swe-pruner-compatible-v1",
                    "model_forward_count": 0,
                    "llm_token_count": 0,
                },
            )
            return
        if path == "/metrics":
            self._send_json(HTTPStatus.OK, self.server.metrics.to_dict())
            return
        if path.startswith("/raw/"):
            raw_id = path.removeprefix("/raw/")
            try:
                text = self.server.raw_store.read(raw_id)
            except KeyError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                return
            self._send_text(HTTPStatus.OK, text)
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path.rstrip("/")
        if path != "/prune":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        started_at = time.perf_counter()
        request: PruningRequest | None = None
        try:
            request = PruningRequest.from_dict(self._read_payload())
            result = self.server.pruner.prune(request)
            self.server.metrics.record(result)
            if self.server.metrics.requests % 100 == 0:
                self.server.raw_store.purge_expired()
            self._send_json(HTTPStatus.OK, result.to_dict())
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - defensive server boundary
            LOGGER.exception("zero-forward pruning request failed")
            if self.server.fail_open and request is not None:
                result = unchanged_result(
                    self.server.pruner.name,
                    request,
                    status="error",
                    started_at=started_at,
                    reason="server-fail-open",
                    error=str(exc),
                )
                self.server.metrics.record(result)
                self._send_json(HTTPStatus.OK, result.to_dict())
            else:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Serve training-free tool-output pruning. This process never calls "
            "vLLM and always reports zero model forwards."
        )
    )
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8121)
    parser.add_argument("--raw-store", default=".zero_forward_raw")
    parser.add_argument("--raw-ttl-hours", type=float, default=72.0)
    parser.add_argument("--public-base-url", default="")
    parser.add_argument("--min-input-tokens", type=int, default=1500)
    parser.add_argument("--min-savings-tokens", type=int, default=256)
    parser.add_argument("--max-retention-ratio", type=float, default=0.85)
    parser.add_argument("--max-cpu-ms", type=float, default=50.0)
    parser.add_argument("--max-output-chars", type=int, default=9000)
    parser.add_argument("--block-max-lines", type=int, default=16)
    parser.add_argument("--fail-closed", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be in [1, 65535]")
    public_base_url = args.public_base_url.strip() or f"http://host.docker.internal:{args.port}"
    raw_store = RawStore(
        Path(args.raw_store),
        ttl_hours=args.raw_ttl_hours,
    )
    raw_store.purge_expired()
    pruner = build_pruner(
        args.method,
        raw_store=raw_store,
        values={
            "min_input_tokens": args.min_input_tokens,
            "min_savings_tokens": args.min_savings_tokens,
            "max_retention_ratio": args.max_retention_ratio,
            "max_cpu_ms": args.max_cpu_ms,
            "max_output_chars": args.max_output_chars,
            "block_max_lines": args.block_max_lines,
            "public_base_url": public_base_url,
        },
    )
    server = ZeroForwardHTTPServer(
        (args.host, args.port),
        pruner=pruner,
        raw_store=raw_store,
        fail_open=not args.fail_closed,
    )
    LOGGER.info(
        "serving method=%s zero_forward=true at http://%s:%d recovery=%s",
        pruner.name,
        args.host,
        args.port,
        public_base_url,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
