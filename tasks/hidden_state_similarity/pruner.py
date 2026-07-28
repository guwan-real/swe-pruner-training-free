from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tf_pruning.protocol import (
    LineScore,
    PruningRequest,
    PruningResult,
)
from tf_pruning.selection import render_pruned_text, select_line_numbers
from tf_pruning.text import error_anchor_lines

ANCHOR_NAMES = ("query", "tool", "error", "decode")
POOLING_ALIASES = {
    "last_token": "last",
    "last-token": "last",
    "last4": "last-4",
    "last_4": "last-4",
    "last4-mean": "last-4",
}


@dataclass(frozen=True)
class HiddenStateSimilarityConfig:
    """Non-learned readout configuration.

    ``mean``, ``max`` and ``last`` use the final hidden-state layer.
    ``last-4`` averages every token representation over the final four
    available layers before producing a line vector.
    """

    pooling: str = "mean"
    anchor_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "query": 0.35,
            "tool": 0.20,
            "error": 0.20,
            "decode": 0.25,
        }
    )
    line_map_base: str = "auto"
    anchor_index_base: int = 0
    derive_error_anchor: bool = True
    expansion_seed_count: int = 0
    show_line_numbers: bool = True
    hidden_states_key: str = "hidden_states"
    token_to_line_key: str = "token_to_line"

    def __post_init__(self) -> None:
        pooling = POOLING_ALIASES.get(self.pooling, self.pooling)
        if pooling not in {"mean", "max", "last", "last-4"}:
            raise ValueError("pooling must be one of: mean, max, last, last-4")
        object.__setattr__(self, "pooling", pooling)
        if self.line_map_base not in {"auto", "zero", "one"}:
            raise ValueError("line_map_base must be auto, zero, or one")
        if self.anchor_index_base not in {0, 1}:
            raise ValueError("anchor_index_base must be 0 or 1")
        if self.expansion_seed_count < 0:
            raise ValueError("expansion_seed_count must be non-negative")
        unknown = set(self.anchor_weights) - set(ANCHOR_NAMES)
        if unknown:
            raise ValueError(f"unknown anchor weights: {sorted(unknown)}")
        if any(float(value) < 0 for value in self.anchor_weights.values()):
            raise ValueError("anchor weights must be non-negative")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
    ) -> "HiddenStateSimilarityConfig":
        if value is None:
            return cls()
        payload = dict(value)
        if "anchor_weights" in payload:
            payload["anchor_weights"] = {
                str(name): float(weight) for name, weight in dict(payload["anchor_weights"]).items()
            }
        return cls(**payload)


def _plain(value: Any) -> Any:
    """Turn optional array-library values into ordinary Python containers."""

    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else value


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _depth(value: Any) -> int:
    value = _plain(value)
    depth = 0
    while _is_sequence(value):
        depth += 1
        if not value:
            break
        value = value[0]
    return depth


def _float_vector(value: Any, *, label: str) -> list[float]:
    value = _plain(value)
    if not _is_sequence(value) or _depth(value) != 1:
        raise ValueError(f"{label} must be a one-dimensional vector")
    result = [float(item) for item in value]
    if not result:
        raise ValueError(f"{label} must not be empty")
    return result


def _mean_vectors(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        raise ValueError("cannot pool an empty vector collection")
    width = len(vectors[0])
    if width == 0 or any(len(vector) != width for vector in vectors):
        raise ValueError("hidden-state vectors must have one fixed width")
    return [
        math.fsum(float(vector[column]) for vector in vectors) / len(vectors)
        for column in range(width)
    ]


def _max_vectors(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        raise ValueError("cannot pool an empty vector collection")
    width = len(vectors[0])
    if width == 0 or any(len(vector) != width for vector in vectors):
        raise ValueError("hidden-state vectors must have one fixed width")
    return [max(float(vector[column]) for vector in vectors) for column in range(width)]


def _as_layers(
    hidden_states: Any,
    *,
    label: str,
) -> list[list[list[float]]]:
    value = _plain(hidden_states)
    depth = _depth(value)
    if depth == 2:
        value = [value]
    elif depth != 3:
        raise ValueError(f"{label} must have shape [tokens, hidden] or [layers, tokens, hidden]")
    layers: list[list[list[float]]] = []
    for layer_no, layer in enumerate(value):
        if not _is_sequence(layer):
            raise ValueError(f"{label} layer {layer_no} is not a sequence")
        layers.append([_float_vector(vector, label=f"{label}[{layer_no}]") for vector in layer])
    if not layers or not layers[-1]:
        raise ValueError(f"{label} must contain at least one token")
    token_count = len(layers[0])
    width = len(layers[0][0])
    if any(len(layer) != token_count for layer in layers):
        raise ValueError(f"{label} layers have different token counts")
    if any(len(vector) != width for layer in layers for vector in layer):
        raise ValueError(f"{label} vectors have different widths")
    return layers


def _pool_token_indices(
    layers: Sequence[Sequence[Sequence[float]]],
    token_indices: Sequence[int],
    pooling: str,
) -> list[float]:
    if not token_indices:
        raise ValueError("cannot pool zero token indices")
    if pooling == "last-4":
        vectors = [
            layers[layer_no][token_no]
            for layer_no in range(max(0, len(layers) - 4), len(layers))
            for token_no in token_indices
        ]
        return _mean_vectors(vectors)
    final_layer = layers[-1]
    if pooling == "last":
        return [float(value) for value in final_layer[token_indices[-1]]]
    vectors = [final_layer[token_no] for token_no in token_indices]
    if pooling == "max":
        return _max_vectors(vectors)
    return _mean_vectors(vectors)


def _pool_external(value: Any, pooling: str, *, label: str) -> list[float]:
    value = _plain(value)
    depth = _depth(value)
    if depth == 1:
        return _float_vector(value, label=label)
    layers = _as_layers(value, label=label)
    return _pool_token_indices(
        layers,
        list(range(len(layers[0]))),
        pooling,
    )


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"anchor and line hidden widths differ: {len(right)} != {len(left)}")
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    value = math.fsum(a * b for a, b in zip(left, right))
    return max(-1.0, min(1.0, value / (left_norm * right_norm)))


def _line_numbers(
    mapping: Any,
    *,
    line_count: int,
    expected_tokens: int,
    base: str,
) -> list[int | None]:
    raw = _plain(mapping)
    if not _is_sequence(raw) or _depth(raw) != 1:
        raise ValueError("token_to_line must be a one-dimensional sequence")
    if len(raw) != expected_tokens:
        raise ValueError(
            "token_to_line length does not match hidden-state token count: "
            f"{len(raw)} != {expected_tokens}"
        )
    integer_values = [int(value) for value in raw]
    if base == "auto":
        non_negative = [value for value in integer_values if value >= 0]
        resolved_base = (
            "zero"
            if 0 in non_negative and non_negative and max(non_negative) <= max(0, line_count - 1)
            else "one"
        )
    else:
        resolved_base = base

    result: list[int | None] = []
    for value in integer_values:
        if value < 0:
            result.append(None)
            continue
        line_no = value + 1 if resolved_base == "zero" else value
        result.append(line_no if 1 <= line_no <= line_count else None)
    return result


def _load_npz(path: str | Path) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("loading NPZ inputs requires the optional 'numpy' dependency") from exc
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"hidden-state NPZ does not exist: {source}")
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


class HiddenStateSimilarityPruner:
    """Rank response lines by cosine similarity to frozen-state anchors."""

    name = "hidden_state_similarity"

    def __init__(
        self,
        config: HiddenStateSimilarityConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self.config = (
            config
            if isinstance(config, HiddenStateSimilarityConfig)
            else HiddenStateSimilarityConfig.from_mapping(config)
        )

    def _inputs(
        self,
        request: PruningRequest,
    ) -> tuple[Any, Any, dict[str, Any], str | None]:
        metadata = request.metadata
        npz_path = _first_present(
            [metadata],
            ("hidden_states_path", "representations_path", "npz_path"),
        )
        archive = _load_npz(str(npz_path)) if npz_path is not None else {}
        sources: list[Mapping[str, Any]] = [metadata, archive]
        hidden_states = _first_present(
            sources,
            (
                "hidden_states",
                "representations",
                self.config.hidden_states_key,
            ),
        )
        mapping = _first_present(
            sources,
            (
                "token_to_line",
                "line_ids",
                self.config.token_to_line_key,
            ),
        )
        if hidden_states is None:
            raise ValueError("request.metadata must provide hidden_states or an NPZ path")
        if mapping is None:
            raise ValueError("request.metadata must provide token_to_line or line_ids")
        return hidden_states, mapping, archive, (str(npz_path) if npz_path is not None else None)

    def _anchor_vectors(
        self,
        request: PruningRequest,
        archive: Mapping[str, Any],
        layers: Sequence[Sequence[Sequence[float]]],
        line_embeddings: Mapping[int, Sequence[float]],
    ) -> dict[str, list[float]]:
        metadata = request.metadata
        anchors_value = metadata.get("anchors", {})
        anchors = anchors_value if isinstance(anchors_value, Mapping) else {}
        archive_anchors_value = archive.get("anchors", {})
        archive_anchors = (
            archive_anchors_value if isinstance(archive_anchors_value, Mapping) else {}
        )
        index_map_value = metadata.get("anchor_token_indices", {})
        index_map = index_map_value if isinstance(index_map_value, Mapping) else {}
        vectors: dict[str, list[float]] = {}
        token_count = len(layers[0])

        for name in ANCHOR_NAMES:
            direct = _first_present(
                [anchors, archive_anchors],
                (name,),
            )
            if direct is None:
                direct = _first_present(
                    [metadata, archive],
                    (
                        f"{name}_anchor",
                        f"anchor_{name}",
                        f"{name}_anchor_hidden_states",
                    ),
                )
            if direct is not None:
                vectors[name] = _pool_external(
                    direct,
                    self.config.pooling,
                    label=f"{name} anchor",
                )
                continue

            indices_value = _first_present(
                [index_map],
                (name,),
            )
            if indices_value is None:
                indices_value = _first_present(
                    [metadata, archive],
                    (
                        f"{name}_anchor_token_indices",
                        f"anchor_{name}_token_indices",
                    ),
                )
            if indices_value is None:
                continue
            indices_raw = _plain(indices_value)
            if not _is_sequence(indices_raw):
                indices_raw = [indices_raw]
            indices = [int(index) - self.config.anchor_index_base for index in indices_raw]
            if any(index < 0 or index >= token_count for index in indices):
                raise ValueError(f"{name} anchor token index is out of range")
            vectors[name] = _pool_token_indices(
                layers,
                sorted(set(indices)),
                self.config.pooling,
            )

        if self.config.derive_error_anchor and "error" not in vectors:
            error_vectors = [
                line_embeddings[line_no]
                for line_no in error_anchor_lines(request.lines)
                if line_no in line_embeddings
            ]
            if error_vectors:
                vectors["error"] = _mean_vectors(error_vectors)
        return vectors

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

        hidden_states, raw_mapping, archive, npz_path = self._inputs(request)
        layers = _as_layers(hidden_states, label="hidden_states")
        mapping = _line_numbers(
            raw_mapping,
            line_count=len(lines),
            expected_tokens=len(layers[0]),
            base=self.config.line_map_base,
        )
        tokens_by_line: dict[int, list[int]] = {}
        for token_no, line_no in enumerate(mapping):
            if line_no is not None:
                tokens_by_line.setdefault(line_no, []).append(token_no)
        line_embeddings = {
            line_no: _pool_token_indices(
                layers,
                token_indices,
                self.config.pooling,
            )
            for line_no, token_indices in tokens_by_line.items()
        }
        anchors = self._anchor_vectors(
            request,
            archive,
            layers,
            line_embeddings,
        )
        active_weights = {
            name: float(self.config.anchor_weights.get(name, 0.0))
            for name in ANCHOR_NAMES
            if name in anchors and self.config.anchor_weights.get(name, 0.0) > 0
        }
        weight_sum = math.fsum(active_weights.values())

        scores: list[float] = []
        line_scores: list[LineScore] = []
        for line_no in range(1, len(lines) + 1):
            embedding = line_embeddings.get(line_no)
            similarities: dict[str, float] = {}
            if embedding is not None and weight_sum > 0:
                similarities = {name: _cosine(embedding, anchors[name]) for name in active_weights}
                score = (
                    math.fsum(active_weights[name] * similarities[name] for name in active_weights)
                    / weight_sum
                )
            else:
                score = 0.0
            reasons = [
                f"{name}_cosine={similarities[name]:.6f}"
                for name in ANCHOR_NAMES
                if name in similarities
            ]
            if embedding is None:
                reasons.append("no_mapped_tokens")
            elif not active_weights:
                reasons.append("no_active_anchors")
            scores.append(score)
            line_scores.append(
                LineScore(
                    line_no=line_no,
                    score=score,
                    reasons=tuple(reasons),
                )
            )

        seed_count = min(self.config.expansion_seed_count, len(lines))
        expansion_seeds = sorted(
            range(1, len(lines) + 1),
            key=lambda line_no: (-scores[line_no - 1], line_no),
        )[:seed_count]
        kept = select_line_numbers(
            scores,
            request.budget,
            expansion_seeds=expansion_seeds,
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
            line_scores=tuple(line_scores),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            metadata={
                "pooling": self.config.pooling,
                "layers_available": len(layers),
                "hidden_width": len(layers[0][0]),
                "mapped_token_count": sum(1 for line_no in mapping if line_no is not None),
                "anchors_used": list(active_weights),
                "anchor_weights_used": active_weights,
                "source": "npz" if npz_path is not None else "metadata",
            },
            request_id=request.request_id,
        )
