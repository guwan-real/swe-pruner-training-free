from __future__ import annotations

import argparse
import json
import logging
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping

from posterior_pruning.methods.common import unchanged_result
from posterior_pruning.protocol import PosteriorPruningRequest
from posterior_pruning.registry import METHODS, build_method
from posterior_pruning.scoring import VLLMActionScorer, VLLMScorerConfig

LOGGER = logging.getLogger("posterior_pruning.http_server")


def _parse_json_object(value: str, *, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


class PosteriorPruningHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        method: Any,
        model_id: str,
        min_chars: int,
        fail_open: bool,
    ) -> None:
        super().__init__(server_address, PosteriorPruningHandler)
        self.posterior_method = method
        self.model_id = model_id
        self.min_chars = min_chars
        self.fail_open = fail_open


class PosteriorPruningHandler(BaseHTTPRequestHandler):
    server: PosteriorPruningHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), format % args)

    def _send(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/health":
            self._send(
                HTTPStatus.OK,
                {
                    "status": "healthy",
                    "service": "posterior-pruning",
                    "method": self.server.posterior_method.name,
                    "model": self.server.model_id,
                    "contract": "post-action-v1",
                },
            )
            return
        self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def _read_payload(self) -> Mapping[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("Content-Length must be an integer") from exc
        if length < 0 or length > 64 * 1024 * 1024:
            raise ValueError("request body size is invalid")
        try:
            payload = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"request body must be valid JSON: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("request body must be a JSON object")
        return payload

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/prune":
            self._send(
                HTTPStatus.CONFLICT,
                {
                    "error": (
                        "this is a post-action service; use /prune-post-action with "
                        "messages, observation_index, and next_action"
                    )
                },
            )
            return
        if self.path.rstrip("/") != "/prune-post-action":
            self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        started_at = time.perf_counter()
        request: PosteriorPruningRequest | None = None
        try:
            request = PosteriorPruningRequest.from_dict(self._read_payload())
            if len(request.observation) < self.server.min_chars:
                result = unchanged_result(
                    self.server.posterior_method.name,
                    request,
                    status="skipped",
                    started_at=started_at,
                    diagnostics={
                        "reason": "below-min-chars",
                        "min_chars": self.server.min_chars,
                    },
                )
            else:
                result = self.server.posterior_method.prune(request)
            self._send(HTTPStatus.OK, result.to_dict())
        except ValueError as exc:
            self._send(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - exercised through integration
            LOGGER.exception("posterior pruning failed")
            if self.server.fail_open and request is not None:
                result = unchanged_result(
                    self.server.posterior_method.name,
                    request,
                    status="error",
                    started_at=started_at,
                    error=str(exc),
                )
                self._send(HTTPStatus.OK, result.to_dict())
            else:
                self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve isolated post-action posterior pruning over HTTP"
    )
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8121)
    parser.add_argument("--vllm-api-base", default="http://127.0.0.1:8015/v1")
    parser.add_argument("--vllm-model", default="")
    parser.add_argument("--vllm-api-key", default="EMPTY")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--min-chars", type=int, default=500)
    parser.add_argument("--max-mean-logprob-drop", type=float, default=0.08)
    parser.add_argument("--block-max-lines", type=int, default=12)
    parser.add_argument("--budget-ratios", default="0.25,0.4,0.6,0.8")
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--max-evaluations", type=int, default=6)
    parser.add_argument("--chat-template-kwargs-json", default="{}")
    parser.add_argument(
        "--fail-closed",
        action="store_true",
        help="Return HTTP 500 instead of the original observation when scoring fails",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 <= args.min_chars:
        raise SystemExit("--min-chars must be non-negative")
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    chat_template_kwargs = _parse_json_object(
        args.chat_template_kwargs_json,
        name="--chat-template-kwargs-json",
    )
    scorer = VLLMActionScorer(
        VLLMScorerConfig(
            api_base=args.vllm_api_base,
            model=args.vllm_model,
            api_key=args.vllm_api_key,
            timeout=args.timeout,
            chat_template_kwargs=chat_template_kwargs,
        )
    )
    model_id = scorer.resolve_model()
    method = build_method(
        args.method,
        scorer,
        {
            "max_mean_logprob_drop": args.max_mean_logprob_drop,
            "block_max_lines": args.block_max_lines,
            "ratios": args.budget_ratios,
            "max_candidates": args.max_candidates,
            "max_evaluations": args.max_evaluations,
            "max_block_evaluations": args.max_evaluations,
        },
    )
    server = PosteriorPruningHTTPServer(
        (args.host, args.port),
        method=method,
        model_id=model_id,
        min_chars=args.min_chars,
        fail_open=not args.fail_closed,
    )
    LOGGER.info(
        "serving method=%s contract=post-action-v1 model=%s at http://%s:%d",
        method.name,
        model_id,
        args.host,
        args.port,
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
