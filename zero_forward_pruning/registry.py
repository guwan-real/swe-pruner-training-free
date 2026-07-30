from __future__ import annotations

from typing import Any, Mapping

from zero_forward_pruning.methods.adaptive_evidence import AdaptiveEvidencePruner
from zero_forward_pruning.methods.common import PrunerConfig
from zero_forward_pruning.methods.intent_ir import IntentIRPruner
from zero_forward_pruning.methods.intent_structure import IntentStructurePruner
from zero_forward_pruning.methods.safe_rules import SafeRulesPruner
from zero_forward_pruning.store import RawStore

METHODS = ("safe_rules", "intent_ir", "intent_structure", "adaptive_evidence")


def build_pruner(
    name: str,
    *,
    raw_store: RawStore,
    values: Mapping[str, Any] | None = None,
):
    values = dict(values or {})
    config = PrunerConfig(
        min_input_tokens=int(values.get("min_input_tokens", 1500)),
        min_savings_tokens=int(values.get("min_savings_tokens", 256)),
        max_retention_ratio=float(values.get("max_retention_ratio", 0.85)),
        max_cpu_ms=float(values.get("max_cpu_ms", 50.0)),
        max_output_chars=int(values.get("max_output_chars", 9000)),
        block_max_lines=int(values.get("block_max_lines", 16)),
        public_base_url=str(values.get("public_base_url", "http://host.docker.internal:8121")),
        raw_store=raw_store,
        require_recovery=bool(values.get("require_recovery", True)),
    )
    classes = {
        "safe_rules": SafeRulesPruner,
        "intent_ir": IntentIRPruner,
        "intent_structure": IntentStructurePruner,
        "adaptive_evidence": AdaptiveEvidencePruner,
    }
    try:
        pruner_class = classes[name]
    except KeyError as exc:
        raise ValueError(f"unknown zero-forward method: {name}") from exc
    return pruner_class(config)
