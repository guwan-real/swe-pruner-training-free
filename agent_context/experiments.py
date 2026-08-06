from __future__ import annotations

import copy
import itertools
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from agent_context.config import AgentContextConfig


def _set_path(payload: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    target = payload
    for part in parts[:-1]:
        child = target.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"cannot set {path!r}; {part!r} is not an object")
        target = child
    target[parts[-1]] = value


def _slug(value: Any) -> str:
    text = str(value).strip().lower()
    return re.sub(r"[^a-z0-9._-]+", "-", text).strip("-") or "value"


@dataclass(frozen=True)
class ExperimentArm:
    name: str
    config: AgentContextConfig
    factors: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "factors": dict(self.factors),
            "config": self.config.to_dict(),
        }


def expand_ablation_matrix(
    base: AgentContextConfig | Mapping[str, Any],
    axes: Mapping[str, Sequence[Any]],
    *,
    prefix: str = "agent-context",
) -> tuple[ExperimentArm, ...]:
    base_config = (
        base if isinstance(base, AgentContextConfig) else AgentContextConfig.from_mapping(base)
    )
    base_payload = base_config.to_dict()
    names = tuple(axes)
    values = tuple(tuple(axes[name]) for name in names)
    if any(not choices for choices in values):
        raise ValueError("ablation axes must contain at least one value")
    arms: list[ExperimentArm] = []
    for combination in itertools.product(*values):
        payload = copy.deepcopy(base_payload)
        factors = dict(zip(names, combination, strict=True))
        for path, value in factors.items():
            _set_path(payload, path, value)
        suffix = "__".join(
            f"{path.replace('.', '-')}={_slug(value)}" for path, value in factors.items()
        )
        arms.append(
            ExperimentArm(
                name=f"{prefix}__{suffix}" if suffix else prefix,
                config=AgentContextConfig.from_mapping(payload),
                factors=factors,
            )
        )
    return tuple(arms)
