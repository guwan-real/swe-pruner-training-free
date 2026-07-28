from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from tf_pruning.protocol import BudgetConfig, Pruner, PruningRequest
from tf_pruning.registry import available_methods, build_pruner
from tf_pruning.selection import render_pruned_text


def _estimated_tokens(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text))


def threshold_to_keep_ratio(threshold: float) -> float:
    """Monotonic compatibility mapping for the official threshold field.

    Training-free rankers use a hard budget rather than calibrated
    probabilities. Higher official thresholds should still retain fewer lines,
    so the compatibility endpoint maps threshold `t` to keep ratio `1-t`.
    Callers should send the extension field `keep_ratio` for exact experiments.
    """

    return max(0.01, min(1.0, 1.0 - float(threshold)))


class CompatPruningService:
    def __init__(
        self,
        pruner: Pruner,
        *,
        fail_open: bool = True,
        show_line_numbers: bool = False,
        default_no_prune_below: int = 20,
    ) -> None:
        self.pruner = pruner
        self.fail_open = fail_open
        self.show_line_numbers = show_line_numbers
        self.default_no_prune_below = default_no_prune_below

    def _budget(self, payload: Mapping[str, Any]) -> BudgetConfig:
        raw_budget = payload.get("budget")
        if isinstance(raw_budget, Mapping):
            return BudgetConfig(**dict(raw_budget))
        keep_ratio = payload.get("keep_ratio")
        if keep_ratio is None:
            keep_ratio = threshold_to_keep_ratio(float(payload.get("threshold", 0.5)))
        return BudgetConfig(
            keep_ratio=float(keep_ratio),
            min_lines=int(payload.get("min_lines", 1)),
            max_lines=(None if payload.get("max_lines") is None else int(payload["max_lines"])),
            no_prune_below=int(
                payload.get(
                    "no_prune_below",
                    self.default_no_prune_below,
                )
            ),
            context_window=int(payload.get("context_window", 1)),
        )

    def _full_response(
        self,
        *,
        code: str,
        query: str,
        error: str,
    ) -> dict[str, Any]:
        lines = code.splitlines()
        token_count = _estimated_tokens(code)
        return {
            "score": 1.0,
            "pruned_code": code,
            "token_scores": [[line, 1.0] for line in lines],
            "kept_frags": list(range(1, len(lines) + 1)),
            "origin_token_cnt": token_count,
            "left_token_cnt": token_count,
            "model_input_token_cnt": token_count + _estimated_tokens(query),
            "error_msg": error,
            "method": getattr(self.pruner, "name", type(self.pruner).__name__),
            "fail_open": True,
            "score_semantics": "fail_open_constant",
            "token_scores_granularity": "line",
        }

    def prune_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        code_value = payload.get("code", payload.get("text"))
        if code_value is None:
            raise ValueError("request must contain code or text")
        code = str(code_value)
        query = str(payload.get("query", ""))
        recent_raw = payload.get("recent_context", ())
        if isinstance(recent_raw, (str, bytes)):
            recent_context = (str(recent_raw),)
        else:
            recent_context = tuple(str(item) for item in recent_raw)
        metadata_raw = payload.get("metadata", {})
        if not isinstance(metadata_raw, Mapping):
            raise ValueError("metadata must be an object")

        request = PruningRequest(
            text=code,
            query=query,
            tool_type=str(payload.get("tool_type", "auto")),
            path=(None if payload.get("path") is None else str(payload["path"])),
            recent_context=recent_context,
            budget=self._budget(payload),
            metadata=dict(metadata_raw),
            request_id=(None if payload.get("request_id") is None else str(payload["request_id"])),
        )
        try:
            result = self.pruner.prune(request)
        except Exception as exc:
            if not self.fail_open:
                raise
            return self._full_response(
                code=code,
                query=query,
                error=f"{type(exc).__name__}: {exc}",
            )

        score_by_line = {score.line_no: score.score for score in result.line_scores}
        kept_scores = [
            score_by_line[line_no]
            for line_no in result.kept_line_numbers
            if line_no in score_by_line
        ]
        pruned_code = render_pruned_text(
            request.lines,
            result.kept_line_numbers,
            show_line_numbers=self.show_line_numbers,
        )
        kept_source = "\n".join(
            request.lines[line_no - 1]
            for line_no in result.kept_line_numbers
            if 1 <= line_no <= len(request.lines)
        )
        return {
            "score": max(kept_scores, default=0.0),
            "pruned_code": pruned_code,
            "token_scores": [
                [line, float(score_by_line.get(line_no, 0.0))]
                for line_no, line in enumerate(request.lines, start=1)
            ],
            "kept_frags": list(result.kept_line_numbers),
            "origin_token_cnt": _estimated_tokens(code),
            "left_token_cnt": _estimated_tokens(kept_source),
            "model_input_token_cnt": (_estimated_tokens(code) + _estimated_tokens(query)),
            "error_msg": None,
            "method": result.method,
            "request_id": result.request_id,
            "latency_ms": result.latency_ms,
            "retention_ratio": result.retention_ratio,
            "line_scores": [line_score.to_dict() for line_score in result.line_scores],
            "score_semantics": "method_native_max_kept_line_score",
            "token_scores_granularity": "line",
        }


def make_handler(
    service: CompatPruningService,
    *,
    max_body_bytes: int = 64 * 1024 * 1024,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _write(self, status: int, payload: Mapping[str, Any]) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._write(404, {"error": "not found"})
                return
            self._write(
                200,
                {
                    "status": "healthy",
                    "method": getattr(
                        service.pruner,
                        "name",
                        type(service.pruner).__name__,
                    ),
                    "training_free": True,
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/prune":
                self._write(404, {"error": "not found"})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length < 0 or content_length > max_body_bytes:
                    raise ValueError("request body is too large")
                raw = self.rfile.read(content_length)
                payload = json.loads(raw.decode("utf-8"))
                if not isinstance(payload, Mapping):
                    raise ValueError("request body must be a JSON object")
                response = service.prune_payload(payload)
            except Exception as exc:
                self._write(
                    400,
                    {"error": f"{type(exc).__name__}: {exc}"},
                )
                return
            self._write(200, response)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def _load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config must contain a JSON object")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Official-shape local HTTP server for training-free pruning"
    )
    parser.add_argument("--method", required=True, choices=available_methods())
    parser.add_argument("--config")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--fail-closed", action="store_true")
    parser.add_argument("--show-line-numbers", action="store_true")
    parser.add_argument("--no-prune-below", type=int, default=20)
    parser.add_argument(
        "--max-body-mb",
        type=int,
        default=64,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be in [1, 65535]")
    if args.max_body_mb < 1:
        raise ValueError("max-body-mb must be positive")
    pruner = build_pruner(args.method, _load_config(args.config))
    service = CompatPruningService(
        pruner,
        fail_open=not args.fail_closed,
        show_line_numbers=args.show_line_numbers,
        default_no_prune_below=args.no_prune_below,
    )
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(
            service,
            max_body_bytes=args.max_body_mb * 1024 * 1024,
        ),
    )
    print(
        f"training-free pruner listening on "
        f"http://{args.host}:{args.port}/prune "
        f"(method={args.method}, fail_open={not args.fail_closed})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
