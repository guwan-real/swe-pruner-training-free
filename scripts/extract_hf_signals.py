#!/usr/bin/env python3
"""Export hidden-state and decode-attention signals from a local HF model.

This is an offline research adapter, not a production serving integration. It
never downloads a model and never updates model parameters.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class PromptLayout:
    prompt: str
    query_start: int
    query_end: int
    tool_start: int
    tool_end: int
    response_start: int
    response_end: int


def build_prompt(request: dict[str, Any]) -> PromptLayout:
    query = str(request.get("query", ""))
    tool_type = str(request.get("tool_type", "auto"))
    path = request.get("path")
    text = str(request.get("text", ""))

    prefix = "Goal:\n"
    query_start = len(prefix)
    prefix += query
    query_end = len(prefix)
    prefix += "\n\n"
    tool_start = len(prefix)
    prefix += f"Tool type: {tool_type}\n"
    if path:
        prefix += f"Path: {path}\n"
    prefix += "Tool response:\n"
    tool_end = len(prefix)
    response_start = len(prefix)
    prompt = prefix + text
    response_end = len(prompt)
    prompt += "\n\nPredict the coding agent's next action:\n"
    return PromptLayout(
        prompt=prompt,
        query_start=query_start,
        query_end=query_end,
        tool_start=tool_start,
        tool_end=tool_end,
        response_start=response_start,
        response_end=response_end,
    )


def token_indices_in_span(
    offsets: Sequence[Sequence[int]],
    start: int,
    end: int,
) -> list[int]:
    return [
        index
        for index, pair in enumerate(offsets)
        if len(pair) == 2
        and int(pair[1]) > start
        and int(pair[0]) < end
        and int(pair[1]) > int(pair[0])
    ]


def response_token_lines(
    offsets: Sequence[Sequence[int]],
    *,
    layout: PromptLayout,
    response: str,
) -> tuple[list[int], list[int]]:
    token_indices = token_indices_in_span(
        offsets,
        layout.response_start,
        layout.response_end,
    )
    line_numbers: list[int] = []
    for token_index in token_indices:
        token_start = max(
            layout.response_start,
            int(offsets[token_index][0]),
        )
        relative_start = token_start - layout.response_start
        line_numbers.append(response.count("\n", 0, relative_start) + 1)
    return token_indices, line_numbers


def _dtype(torch: Any, name: str) -> Any:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def extract_signals(
    request: dict[str, Any],
    *,
    model_path: str,
    output_path: str | Path,
    output_request_path: str | Path | None,
    device: str,
    dtype: str,
    decode_steps: int,
    hidden_layers: int,
    max_length: int | None,
    trust_remote_code: bool,
    signals: str,
) -> None:
    try:
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "signal extraction needs the model extra: python -m pip install -e '.[model]'"
        ) from exc

    if decode_steps < 1:
        raise ValueError("decode_steps must be positive")
    if hidden_layers < 1:
        raise ValueError("hidden_layers must be positive")

    resolved_device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("the signal extractor requires a fast tokenizer with offset mapping")
    model_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": trust_remote_code,
        "torch_dtype": _dtype(torch, dtype),
    }
    if signals in {"attention", "both"}:
        model_kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()
    model.requires_grad_(False)
    model.to(resolved_device)

    layout = build_prompt(request)
    encoded = tokenizer(
        layout.prompt,
        add_special_tokens=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    sequence_length = int(encoded["input_ids"].shape[1])
    effective_limit = max_length
    if effective_limit is None:
        tokenizer_limit = getattr(tokenizer, "model_max_length", None)
        effective_limit = (
            int(tokenizer_limit)
            if isinstance(tokenizer_limit, int) and 2 <= tokenizer_limit < 1_000_000
            else None
        )
    if effective_limit is not None and sequence_length + decode_steps > effective_limit:
        raise ValueError(
            f"prompt has {sequence_length} tokens plus {decode_steps} decode "
            f"steps, exceeding max_length={effective_limit}; chunk the "
            "observation instead of silently truncating it"
        )

    response = str(request.get("text", ""))
    response_indices, token_to_line = response_token_lines(
        offsets,
        layout=layout,
        response=response,
    )
    if not response_indices:
        raise ValueError("tokenizer produced no tokens for the tool response")
    query_indices = token_indices_in_span(
        offsets,
        layout.query_start,
        layout.query_end,
    )
    tool_indices = token_indices_in_span(
        offsets,
        layout.tool_start,
        layout.tool_end,
    )

    tensors = {name: value.to(resolved_device) for name, value in encoded.items()}
    with torch.inference_mode():
        prefill = model(
            **tensors,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
            return_dict=True,
        )

    layer_count = min(hidden_layers, len(prefill.hidden_states))
    chosen_hidden = prefill.hidden_states[-layer_count:]
    output: dict[str, Any] = {
        "format_version": np.asarray([1], dtype=np.int64),
        "token_to_line": np.asarray(token_to_line, dtype=np.int64),
    }
    if signals in {"hidden", "both"}:
        output["hidden_states"] = (
            torch.stack([layer[0, response_indices, :] for layer in chosen_hidden])
            .float()
            .cpu()
            .numpy()
        )
        if query_indices:
            output["query_anchor"] = (
                torch.stack([layer[0, query_indices, :] for layer in chosen_hidden])
                .float()
                .cpu()
                .numpy()
            )
        if tool_indices:
            output["tool_anchor"] = (
                torch.stack([layer[0, tool_indices, :] for layer in chosen_hidden])
                .float()
                .cpu()
                .numpy()
            )

    generated_ids = tensors["input_ids"]
    generated_mask = tensors.get("attention_mask")
    next_token = prefill.logits[:, -1:, :].argmax(dim=-1)
    decode_hidden_by_step: list[Any] = []
    attention_by_step: list[Any] = []
    for _step in range(decode_steps):
        generated_ids = torch.cat((generated_ids, next_token), dim=1)
        if generated_mask is not None:
            generated_mask = torch.cat(
                (
                    generated_mask,
                    torch.ones(
                        (generated_mask.shape[0], 1),
                        dtype=generated_mask.dtype,
                        device=generated_mask.device,
                    ),
                ),
                dim=1,
            )
        with torch.inference_mode():
            decoded = model(
                input_ids=generated_ids,
                attention_mask=generated_mask,
                output_hidden_states=True,
                output_attentions=signals in {"attention", "both"},
                use_cache=False,
                return_dict=True,
            )
        decode_layers = decoded.hidden_states[-layer_count:]
        decode_hidden_by_step.append(torch.stack([layer[0, -1, :] for layer in decode_layers]))
        if signals in {"attention", "both"}:
            if not decoded.attentions:
                raise RuntimeError(
                    "model did not return attentions; verify that eager attention is supported"
                )
            attention_by_step.append(
                torch.stack(
                    [
                        layer[0, :, -1, response_indices]
                        for layer in decoded.attentions[-layer_count:]
                    ]
                )
            )
        next_token = decoded.logits[:, -1:, :].argmax(dim=-1)

    if signals in {"hidden", "both"}:
        output["decode_anchor"] = torch.stack(decode_hidden_by_step, dim=1).float().cpu().numpy()
    if signals in {"attention", "both"}:
        output["attention"] = torch.stack(attention_by_step, dim=2).float().cpu().numpy()

    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **output)

    companion = (
        Path(output_request_path).expanduser().resolve()
        if output_request_path
        else target.with_suffix(".request.json")
    )
    request_copy = dict(request)
    metadata = dict(request_copy.get("metadata", {}))
    if signals in {"hidden", "both"}:
        metadata["hidden_states_path"] = str(target)
    if signals in {"attention", "both"}:
        metadata["attention_path"] = str(target)
    request_copy["metadata"] = metadata
    companion.write_text(
        json.dumps(request_copy, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "signals": signals,
                "npz": str(target),
                "request": str(companion),
                "response_tokens": len(response_indices),
                "hidden_layers": layer_count,
                "decode_steps": decode_steps,
                "device": resolved_device,
                "training": False,
                "local_files_only": True,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export local frozen-model hidden/attention NPZ signals"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-request")
    parser.add_argument(
        "--signals",
        choices=("hidden", "attention", "both"),
        default="both",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--decode-steps", type=int, default=3)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.request).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--request must contain a JSON object")
    request = payload.get("request", payload)
    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object")
    extract_signals(
        request,
        model_path=args.model_path,
        output_path=args.output,
        output_request_path=args.output_request,
        device=args.device,
        dtype=args.dtype,
        decode_steps=args.decode_steps,
        hidden_layers=args.hidden_layers,
        max_length=args.max_length,
        trust_remote_code=args.trust_remote_code,
        signals=args.signals,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
