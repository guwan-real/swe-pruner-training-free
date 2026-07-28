from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tf_pruning.protocol import (
    BudgetConfig,
    LineScore,
    PruningRequest,
    PruningResult,
)
from tf_pruning.selection import render_pruned_text, select_line_numbers
from tf_pruning.text import structural_anchor_lines

AXIS_ALIASES = {
    "layer": "layers",
    "head": "heads",
    "step": "steps",
    "decode": "steps",
    "decode_steps": "steps",
    "token": "tokens",
    "key": "keys",
    "query": "queries",
}


@dataclass(frozen=True)
class AttentionPrunerConfig:
    """Configuration for attention-mass and matrix-rollout readouts."""

    method: str = "attention_mass"
    attention_layout: str | None = None
    layers: Any = "last-4"
    heads: Any = "all"
    decode_steps: Any = "first-3"
    layer_aggregation: str = "mean"
    head_aggregation: str = "mean"
    step_aggregation: str = "mean"
    step_decay: float = 0.8
    line_aggregation: str = "sum"
    sink_first_tokens: int = 0
    sink_token_indices: tuple[int, ...] = ()
    top_p: float = 0.9
    selection_mode: str = "hybrid"
    structure_floor: float = 0.08
    local_floor: float = 0.04
    local_window: int = 1
    local_seed_count: int = 3
    rollout_residual_weight: float = 1.0
    line_map_base: str = "auto"
    batch_index: int = 0
    show_line_numbers: bool = True
    attention_key: str = "attention"
    token_to_line_key: str = "token_to_line"

    def __post_init__(self) -> None:
        if self.method not in {"attention_mass", "rollout"}:
            raise ValueError("method must be attention_mass or rollout")
        valid_aggregations = {"mean", "max", "sum"}
        if self.layer_aggregation not in valid_aggregations:
            raise ValueError("invalid layer_aggregation")
        if self.head_aggregation not in valid_aggregations:
            raise ValueError("invalid head_aggregation")
        if self.step_aggregation not in valid_aggregations | {"weighted"}:
            raise ValueError("invalid step_aggregation")
        if self.line_aggregation not in valid_aggregations:
            raise ValueError("invalid line_aggregation")
        if not 0.0 < self.step_decay <= 1.0:
            raise ValueError("step_decay must be in (0, 1]")
        if self.sink_first_tokens < 0:
            raise ValueError("sink_first_tokens must be non-negative")
        if any(index < 0 for index in self.sink_token_indices):
            raise ValueError("sink_token_indices must be zero-based and non-negative")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.selection_mode not in {"hard_budget", "top_p", "hybrid"}:
            raise ValueError("selection_mode must be hard_budget, top_p, or hybrid")
        if not 0.0 <= self.structure_floor <= 1.0:
            raise ValueError("structure_floor must be in [0, 1]")
        if not 0.0 <= self.local_floor <= 1.0:
            raise ValueError("local_floor must be in [0, 1]")
        if self.local_window < 0 or self.local_seed_count < 0:
            raise ValueError("local window/count must be non-negative")
        if self.rollout_residual_weight < 0:
            raise ValueError("rollout_residual_weight must be non-negative")
        if self.line_map_base not in {"auto", "zero", "one"}:
            raise ValueError("line_map_base must be auto, zero, or one")
        if self.batch_index < 0:
            raise ValueError("batch_index must be non-negative")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
    ) -> "AttentionPrunerConfig":
        if value is None:
            return cls()
        payload = dict(value)
        if "sink_token_indices" in payload:
            payload["sink_token_indices"] = tuple(
                int(index) for index in payload["sink_token_indices"]
            )
        return cls(**payload)


def _plain(value: Any) -> Any:
    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else value


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _shape(value: Any, *, label: str) -> tuple[int, ...]:
    value = _plain(value)
    if not _is_sequence(value):
        return ()
    size = len(value)
    if size == 0:
        return (0,)
    child_shape = _shape(value[0], label=label)
    for child in value[1:]:
        if _shape(child, label=label) != child_shape:
            raise ValueError(f"{label} must be a rectangular tensor")
    return (size, *child_shape)


def _at(value: Any, indices: Sequence[int]) -> float:
    current = value
    for index in indices:
        current = current[index]
    return float(current)


def _layout(
    configured: str | None,
    *,
    ndim: int,
    method: str,
) -> tuple[str, ...]:
    if configured is None:
        if method == "rollout":
            defaults = {
                3: ("layers", "queries", "keys"),
                4: ("layers", "heads", "queries", "keys"),
                5: ("batch", "layers", "heads", "queries", "keys"),
            }
        else:
            defaults = {
                1: ("tokens",),
                2: ("steps", "tokens"),
                3: ("heads", "steps", "tokens"),
                4: ("layers", "heads", "steps", "tokens"),
                5: ("batch", "layers", "heads", "steps", "tokens"),
            }
        if ndim not in defaults:
            raise ValueError(f"cannot infer attention layout for a {ndim}-D tensor")
        return defaults[ndim]
    raw = configured.replace(",", " ").split()
    axes = tuple(AXIS_ALIASES.get(axis.strip(), axis.strip()) for axis in raw)
    if len(axes) != ndim:
        raise ValueError(f"attention_layout has {len(axes)} axes but tensor has {ndim}")
    if len(set(axes)) != len(axes):
        raise ValueError("attention_layout axes must be unique")
    return axes


def _indices(
    selector: Any,
    size: int,
    *,
    label: str,
) -> list[int]:
    if size <= 0:
        raise ValueError(f"{label} axis must not be empty")
    selector = _plain(selector)
    if isinstance(selector, str):
        value = selector.lower()
        if value == "all":
            result = list(range(size))
        elif value == "first":
            result = [0]
        elif value == "last":
            result = [size - 1]
        elif value.startswith("first-"):
            count = int(value.split("-", 1)[1])
            result = list(range(min(size, count)))
        elif value.startswith("last-"):
            count = int(value.split("-", 1)[1])
            result = list(range(max(0, size - count), size))
        else:
            result = [int(part) for part in value.split(",") if part]
    elif _is_sequence(selector):
        result = [int(index) for index in selector]
    else:
        result = [int(selector)]
    normalized = [index + size if index < 0 else index for index in result]
    if not normalized or any(index < 0 or index >= size for index in normalized):
        raise ValueError(f"{label} selector contains an out-of-range index")
    return sorted(set(normalized))


def _aggregate(
    values: Sequence[float],
    mode: str,
    *,
    decay: float = 1.0,
) -> float:
    if not values:
        return 0.0
    if mode == "max":
        return max(values)
    if mode == "sum":
        return math.fsum(values)
    if mode == "weighted":
        weights = [decay**index for index in range(len(values))]
        return math.fsum(value * weight for value, weight in zip(values, weights)) / math.fsum(
            weights
        )
    return math.fsum(values) / len(values)


def _coords(
    axes: Sequence[str],
    values: Mapping[str, int],
) -> list[int]:
    return [values.get(axis, 0) for axis in axes]


def _attention_mass(
    tensor: Any,
    shape: Sequence[int],
    axes: Sequence[str],
    config: AttentionPrunerConfig,
    *,
    decode_override: Any | None,
) -> tuple[list[float], dict[str, list[int]]]:
    if "tokens" not in axes:
        raise ValueError("attention_mass layout must contain a tokens axis")
    sizes = dict(zip(axes, shape))
    batch = config.batch_index
    if "batch" in sizes and batch >= sizes["batch"]:
        raise ValueError("batch_index is out of range")
    layers = _indices(
        config.layers,
        sizes.get("layers", 1),
        label="layers",
    )
    heads = _indices(config.heads, sizes.get("heads", 1), label="heads")
    steps = _indices(
        decode_override if decode_override is not None else config.decode_steps,
        sizes.get("steps", 1),
        label="decode_steps",
    )

    scores: list[float] = []
    for token in range(sizes["tokens"]):
        layer_values: list[float] = []
        for layer in layers:
            head_values: list[float] = []
            for head in heads:
                step_values = [
                    max(
                        0.0,
                        _at(
                            tensor,
                            _coords(
                                axes,
                                {
                                    "batch": batch,
                                    "layers": layer,
                                    "heads": head,
                                    "steps": step,
                                    "tokens": token,
                                },
                            ),
                        ),
                    )
                    for step in steps
                ]
                head_values.append(
                    _aggregate(
                        step_values,
                        config.step_aggregation,
                        decay=config.step_decay,
                    )
                )
            layer_values.append(_aggregate(head_values, config.head_aggregation))
        scores.append(_aggregate(layer_values, config.layer_aggregation))
    return scores, {"layers": layers, "heads": heads, "decode_steps": steps}


def _identity(size: int) -> list[list[float]]:
    return [[1.0 if row == column else 0.0 for column in range(size)] for row in range(size)]


def _matmul(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
) -> list[list[float]]:
    size = len(left)
    return [
        [
            math.fsum(left[row][mid] * right[mid][column] for mid in range(size))
            for column in range(size)
        ]
        for row in range(size)
    ]


def _rollout(
    tensor: Any,
    shape: Sequence[int],
    axes: Sequence[str],
    config: AttentionPrunerConfig,
    *,
    decode_override: Any | None,
) -> tuple[list[float], dict[str, list[int]]]:
    if "queries" not in axes or "keys" not in axes:
        raise ValueError("rollout layout must contain queries and keys axes")
    sizes = dict(zip(axes, shape))
    query_count = sizes["queries"]
    key_count = sizes["keys"]
    if query_count != key_count:
        raise ValueError("rollout requires square query/key attention matrices")
    batch = config.batch_index
    if "batch" in sizes and batch >= sizes["batch"]:
        raise ValueError("batch_index is out of range")
    layers = _indices(
        config.layers,
        sizes.get("layers", 1),
        label="layers",
    )
    heads = _indices(config.heads, sizes.get("heads", 1), label="heads")
    queries = _indices(
        decode_override if decode_override is not None else config.decode_steps,
        query_count,
        label="decode_steps",
    )
    joint = _identity(key_count)
    for layer in layers:
        matrix: list[list[float]] = []
        for query in range(query_count):
            row: list[float] = []
            for key in range(key_count):
                values = [
                    max(
                        0.0,
                        _at(
                            tensor,
                            _coords(
                                axes,
                                {
                                    "batch": batch,
                                    "layers": layer,
                                    "heads": head,
                                    "queries": query,
                                    "keys": key,
                                },
                            ),
                        ),
                    )
                    for head in heads
                ]
                value = _aggregate(values, config.head_aggregation)
                if query == key:
                    value += config.rollout_residual_weight
                row.append(value)
            denominator = math.fsum(row)
            matrix.append(
                [value / denominator for value in row]
                if denominator > 0
                else _identity(key_count)[query]
            )
        joint = _matmul(matrix, joint)
    token_scores = [
        _aggregate(
            [joint[query][key] for query in queries],
            config.step_aggregation,
            decay=config.step_decay,
        )
        for key in range(key_count)
    ]
    return token_scores, {
        "layers": layers,
        "heads": heads,
        "decode_steps": queries,
    }


def _line_numbers(
    mapping: Any,
    *,
    line_count: int,
    expected_tokens: int,
    base: str,
) -> list[int | None]:
    raw = _plain(mapping)
    if not _is_sequence(raw) or len(_shape(raw, label="token_to_line")) != 1:
        raise ValueError("token_to_line must be a one-dimensional sequence")
    if len(raw) != expected_tokens:
        raise ValueError(
            "token_to_line length does not match attention token count: "
            f"{len(raw)} != {expected_tokens}"
        )
    values = [int(value) for value in raw]
    if base == "auto":
        non_negative = [value for value in values if value >= 0]
        resolved = (
            "zero"
            if 0 in non_negative and non_negative and max(non_negative) <= max(0, line_count - 1)
            else "one"
        )
    else:
        resolved = base
    result: list[int | None] = []
    for value in values:
        if value < 0:
            result.append(None)
            continue
        line_no = value + 1 if resolved == "zero" else value
        result.append(line_no if 1 <= line_no <= line_count else None)
    return result


def _load_npz(path: str | Path) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("loading NPZ inputs requires the optional 'numpy' dependency") from exc
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"attention NPZ does not exist: {source}")
    with np.load(source, allow_pickle=False) as archive:
        return {name: archive[name].tolist() for name in archive.files}


def _first_present(
    mappings: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> Any | None:
    for mapping in mappings:
        for key in keys:
            if key in mapping and mapping[key] is not None:
                return mapping[key]
    return None


def _nucleus(scores: Sequence[float], top_p: float) -> list[int]:
    ranked = sorted(
        range(1, len(scores) + 1),
        key=lambda line_no: (-scores[line_no - 1], line_no),
    )
    total = math.fsum(max(0.0, score) for score in scores)
    if not ranked:
        return []
    if total <= 0.0:
        return [ranked[0]]
    selected: list[int] = []
    cumulative = 0.0
    for line_no in ranked:
        selected.append(line_no)
        cumulative += max(0.0, scores[line_no - 1])
        if cumulative / total >= top_p:
            break
    return selected


class AttentionRolloutPruner:
    """Aggregate frozen-model attention and apply a shared line budget."""

    name = "attention_rollout"

    def __init__(
        self,
        config: AttentionPrunerConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self.config = (
            config
            if isinstance(config, AttentionPrunerConfig)
            else AttentionPrunerConfig.from_mapping(config)
        )

    def _inputs(
        self,
        request: PruningRequest,
    ) -> tuple[Any, Any, str | None]:
        metadata = request.metadata
        npz_path = _first_present(
            [metadata],
            ("attention_path", "attentions_path", "npz_path"),
        )
        archive = _load_npz(str(npz_path)) if npz_path is not None else {}
        sources: list[Mapping[str, Any]] = [metadata, archive]
        attention = _first_present(
            sources,
            ("attention", "attentions", "attention_weights", self.config.attention_key),
        )
        mapping = _first_present(
            sources,
            ("token_to_line", "line_ids", self.config.token_to_line_key),
        )
        if attention is None:
            raise ValueError("request.metadata must provide attention or an NPZ path")
        if mapping is None:
            raise ValueError("request.metadata must provide token_to_line or line_ids")
        return attention, mapping, (str(npz_path) if npz_path is not None else None)

    def prune(self, request: PruningRequest) -> PruningResult:
        started = time.perf_counter()
        lines = request.lines
        if not lines:
            return PruningResult(
                method=self.name,
                original_line_count=0,
                kept_line_numbers=(),
                pruned_text="",
                latency_ms=(time.perf_counter() - started) * 1000.0,
                request_id=request.request_id,
            )
        attention, raw_mapping, npz_path = self._inputs(request)
        attention = _plain(attention)
        shape = _shape(attention, label="attention")
        axes = _layout(
            self.config.attention_layout,
            ndim=len(shape),
            method=self.config.method,
        )
        decode_override = _first_present(
            [request.metadata],
            (
                ("decode_token_indices", "decode_step_indices")
                if self.config.method == "rollout"
                else ("decode_step_indices",)
            ),
        )
        if self.config.method == "rollout":
            token_scores, selected_axes = _rollout(
                attention,
                shape,
                axes,
                self.config,
                decode_override=decode_override,
            )
        else:
            token_scores, selected_axes = _attention_mass(
                attention,
                shape,
                axes,
                self.config,
                decode_override=decode_override,
            )

        mapping = _line_numbers(
            raw_mapping,
            line_count=len(lines),
            expected_tokens=len(token_scores),
            base=self.config.line_map_base,
        )
        dynamic_sinks = _first_present(
            [request.metadata],
            ("sink_token_indices", "attention_sink_indices"),
        )
        sink_indices = set(self.config.sink_token_indices)
        sink_indices.update(range(min(self.config.sink_first_tokens, len(token_scores))))
        if dynamic_sinks is not None:
            raw_sinks = _plain(dynamic_sinks)
            if not _is_sequence(raw_sinks):
                raw_sinks = [raw_sinks]
            sink_indices.update(int(index) for index in raw_sinks)
        sink_indices = {index for index in sink_indices if 0 <= index < len(token_scores)}

        tokens_by_line: dict[int, list[float]] = {}
        for token_no, (line_no, score) in enumerate(zip(mapping, token_scores)):
            if line_no is not None and token_no not in sink_indices:
                tokens_by_line.setdefault(line_no, []).append(max(0.0, score))
        raw_line_scores = [
            _aggregate(
                tokens_by_line.get(line_no, ()),
                self.config.line_aggregation,
            )
            for line_no in range(1, len(lines) + 1)
        ]
        maximum = max(raw_line_scores, default=0.0)
        scores = [score / maximum if maximum > 0.0 else 0.0 for score in raw_line_scores]
        reasons: dict[int, list[str]] = {
            line_no: [f"attention={raw_line_scores[line_no - 1]:.8f}"]
            for line_no in range(1, len(lines) + 1)
        }

        structural = structural_anchor_lines(lines)
        for line_no in structural:
            if scores[line_no - 1] < self.config.structure_floor:
                scores[line_no - 1] = self.config.structure_floor
                reasons[line_no].append(f"structure_floor={self.config.structure_floor:.6f}")

        local_seeds = sorted(
            range(1, len(lines) + 1),
            key=lambda line_no: (-scores[line_no - 1], line_no),
        )[: min(self.config.local_seed_count, len(lines))]
        for seed in local_seeds:
            start = max(1, seed - self.config.local_window)
            end = min(len(lines), seed + self.config.local_window)
            for line_no in range(start, end + 1):
                if scores[line_no - 1] < self.config.local_floor:
                    scores[line_no - 1] = self.config.local_floor
                    reasons[line_no].append(f"local_floor_from={seed}")

        nucleus = _nucleus(scores, self.config.top_p)
        target = request.budget.target_lines(len(lines))
        if len(lines) <= request.budget.no_prune_below:
            kept = select_line_numbers(scores, request.budget)
        elif self.config.selection_mode == "hard_budget":
            kept = select_line_numbers(scores, request.budget)
        elif self.config.selection_mode == "hybrid":
            kept = select_line_numbers(
                scores,
                request.budget,
                expansion_seeds=nucleus,
            )
        else:
            nucleus_count = min(len(nucleus), target)
            if nucleus_count == 0:
                kept = ()
            else:
                nucleus_budget = BudgetConfig(
                    keep_ratio=1.0,
                    min_lines=nucleus_count,
                    max_lines=nucleus_count,
                    no_prune_below=0,
                    context_window=request.budget.context_window,
                )
                kept = select_line_numbers(
                    scores,
                    nucleus_budget,
                    expansion_seeds=nucleus,
                )

        line_score_records = tuple(
            LineScore(
                line_no=line_no,
                score=scores[line_no - 1],
                reasons=tuple(reasons[line_no]),
            )
            for line_no in range(1, len(lines) + 1)
        )
        return PruningResult(
            method=self.name,
            original_line_count=len(lines),
            kept_line_numbers=kept,
            pruned_text=render_pruned_text(
                lines,
                kept,
                show_line_numbers=self.config.show_line_numbers,
            ),
            line_scores=line_score_records,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            metadata={
                "attention_method": self.config.method,
                "attention_layout": list(axes),
                "attention_shape": list(shape),
                "selected_axes": selected_axes,
                "sink_token_indices": sorted(sink_indices),
                "top_p": self.config.top_p,
                "top_p_line_numbers": nucleus,
                "selection_mode": self.config.selection_mode,
                "structural_line_numbers": sorted(structural),
                "local_seed_line_numbers": local_seeds,
                "source": "npz" if npz_path is not None else "metadata",
            },
            request_id=request.request_id,
        )
