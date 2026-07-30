from __future__ import annotations

import argparse
import json

from posterior_pruning.scoring import VLLMActionScorer, VLLMScorerConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe the exact vLLM features required by posterior pruning"
    )
    parser.add_argument("--vllm-api-base", default="http://127.0.0.1:8015/v1")
    parser.add_argument("--vllm-model", default="")
    parser.add_argument("--vllm-api-key", default="EMPTY")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--chat-template-kwargs-json", default="{}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    kwargs = json.loads(args.chat_template_kwargs_json)
    if not isinstance(kwargs, dict):
        raise SystemExit("--chat-template-kwargs-json must be an object")
    scorer = VLLMActionScorer(
        VLLMScorerConfig(
            api_base=args.vllm_api_base,
            model=args.vllm_model,
            api_key=args.vllm_api_key,
            timeout=args.timeout,
            chat_template_kwargs=kwargs,
        )
    )
    score = scorer.score(
        [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Reply with one shell command that prints the path."},
        ],
        "```bash\npwd\n```",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "model": scorer.resolve_model(),
                "scorer": scorer.name,
                "mean_logprob": score.mean_logprob,
                "target_tokens": score.target_tokens,
                "prompt_tokens": score.prompt_tokens,
                "required_endpoints": ["/v1/models", "/v1/tokenize", "/v1/chat/completions"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
