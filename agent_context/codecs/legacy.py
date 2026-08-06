from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from agent_context.codecs.base import (
    CodecRegistry,
    ObservationCodec,
    ViewGenerationConfig,
    full_view,
)
from agent_context.codecs.typed import build_typed_codec_registry
from agent_context.estimation import TokenEstimator
from agent_context.models import (
    ContextSignal,
    ContextView,
    EvidenceDocument,
    Observation,
    ObservationKind,
    ViewLevel,
    omitted_ranges,
)
from agent_context.signals import SignalMatch
from posterior_history_pruning.protocol import PosteriorHistoryConfig, PosteriorSignal
from posterior_history_pruning.selection import compact_after_followup


def _latest_signal(signals: Sequence[ContextSignal], provider: str) -> str:
    return next(
        (signal.text for signal in reversed(signals) if signal.provider == provider),
        "",
    )


@dataclass(frozen=True)
class LegacyPosteriorCodec:
    """Expose the existing binary selector as one framework view candidate."""

    kind: ObservationKind
    delegate: ObservationCodec
    legacy_config: PosteriorHistoryConfig

    @property
    def name(self) -> str:
        return f"legacy_posterior_v1:{self.delegate.name}"

    def parse(self, observation: Observation) -> EvidenceDocument:
        document = self.delegate.parse(observation)
        return EvidenceDocument(
            observation_id=document.observation_id,
            kind=document.kind,
            lines=document.lines,
            units=document.units,
            codec=self.name,
            metadata=document.metadata,
        )

    def generate_views(
        self,
        observation: Observation,
        document: EvidenceDocument,
        signals: Sequence[ContextSignal],
        match: SignalMatch,
        estimator: TokenEstimator,
        config: ViewGenerationConfig,
    ) -> tuple[ContextView, ...]:
        del match, config
        full = full_view(observation, document, estimator)
        posterior = PosteriorSignal(
            command=_latest_signal(signals, "next_action.command"),
            context_focus_question=_latest_signal(signals, "next_action.focus"),
            response_content=_latest_signal(signals, "next_action.response"),
        )
        result = compact_after_followup(
            observation.visible_content,
            causing_command=observation.causing_action,
            causing_path=observation.path,
            posterior=posterior,
            config=self.legacy_config,
        )
        if result.status != "compacted":
            return (full,)
        retained = set(result.retained_line_numbers)
        preserved_units = tuple(
            unit.id for unit in document.units if retained.intersection(unit.line_numbers)
        )
        focused = ContextView(
            observation_id=observation.id,
            level=ViewLevel.FOCUSED,
            content=result.text,
            token_count=result.left_token_cnt,
            codec=self.name,
            policy=result.method,
            preserved_unit_ids=preserved_units,
            omitted_line_ranges=omitted_ranges(len(document.lines), sorted(retained)),
            relevance_score=float(result.matched_block_count),
            reason=result.reason,
            metadata={
                "legacy_status": result.status,
                "legacy_output_kind": result.output_kind,
                "legacy_block_count": result.block_count,
                "legacy_hard_block_count": result.hard_block_count,
                "legacy_matched_block_count": result.matched_block_count,
                "legacy_selected_block_count": result.selected_block_count,
            },
        )
        return tuple(sorted((focused, full), key=lambda view: view.level))


def build_legacy_posterior_codec_registry(options: dict[str, object]) -> CodecRegistry:
    block_max_lines = int(options.get("block_max_lines", 16))
    legacy_config = PosteriorHistoryConfig(
        hot_observations=1,
        min_input_tokens=int(options.get("min_input_tokens", 1500)),
        min_savings_tokens=int(options.get("min_savings_tokens", 256)),
        max_retention_ratio=float(options.get("max_retention_ratio", 0.85)),
        block_max_lines=block_max_lines,
        max_output_chars=int(options.get("max_output_chars", 9000)),
        method=str(options.get("method", "adaptive")),
    )
    delegates = build_typed_codec_registry(block_max_lines=block_max_lines)
    registry = CodecRegistry()
    for kind in ObservationKind:
        registry.register(
            LegacyPosteriorCodec(
                kind=kind,
                delegate=delegates.get(kind),
                legacy_config=legacy_config,
            )
        )
    return registry
