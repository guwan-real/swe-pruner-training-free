from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from .protocol import Pruner

METHOD_MODULES: dict[str, str] = {
    "ir_structural": "tasks.ir_structural",
    "conditional_ppl": "tasks.conditional_ppl",
    "hidden_state_similarity": "tasks.hidden_state_similarity",
    "attention_rollout": "tasks.attention_rollout",
    "influence_oracle": "tasks.influence_oracle",
    "execution_ast": "tasks.execution_ast",
    "ir_ast_hybrid": "tasks.ir_ast_hybrid",
}

ALIASES: dict[str, str] = {
    "ir": "ir_structural",
    "ppl": "conditional_ppl",
    "hidden": "hidden_state_similarity",
    "attention": "attention_rollout",
    "influence": "influence_oracle",
    "ast": "execution_ast",
    "hybrid": "ir_ast_hybrid",
}


def canonical_method_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    return ALIASES.get(normalized, normalized)


def available_methods() -> tuple[str, ...]:
    return tuple(METHOD_MODULES)


def build_pruner(
    name: str,
    config: Mapping[str, Any] | None = None,
) -> Pruner:
    canonical = canonical_method_name(name)
    if canonical not in METHOD_MODULES:
        choices = ", ".join(available_methods())
        raise ValueError(f"unknown method {name!r}; choose one of: {choices}")
    module = importlib.import_module(METHOD_MODULES[canonical])
    factory = getattr(module, "build_pruner", None)
    if factory is None:
        raise RuntimeError(f"{METHOD_MODULES[canonical]} does not export build_pruner(config)")
    payload = dict(config or {})
    method_section = payload.get(canonical)
    if isinstance(method_section, Mapping):
        payload = dict(method_section)
    elif (
        canonical in {"hidden_state_similarity", "attention_rollout"}
        and set(payload) == {"pruner"}
        and isinstance(payload.get("pruner"), Mapping)
    ):
        # These signal-only tasks keep their example settings under `pruner`;
        # unlike PPL/influence, they have no scorer/objective sibling sections.
        payload = dict(payload["pruner"])
    pruner = factory(payload)
    if not hasattr(pruner, "prune"):
        raise TypeError(f"factory for {canonical} did not return a Pruner")
    return pruner
