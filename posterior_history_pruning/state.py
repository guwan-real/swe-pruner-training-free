from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from posterior_history_pruning.protocol import (
    CompactionResult,
    PosteriorHistoryConfig,
    PosteriorSignal,
)
from posterior_history_pruning.selection import compact_after_followup


@dataclass
class ObservationRecord:
    """A canonical observation plus a separately rendered prompt view."""

    message: dict[str, Any]
    raw_output: str
    prefix: str
    suffix: str
    causing_command: str
    causing_path: str
    posterior: PosteriorSignal | None = None
    result: CompactionResult | None = None
    prompt_compaction_count: int = 0
    total_prompt_tokens_saved: int = 0


@dataclass
class PosteriorHistoryState:
    """Per-agent state that never changes canonical observation content.

    ``agent.messages`` remains the full, auditable trajectory.  ``render``
    returns a copied message list only for the next model request.
    """

    config: PosteriorHistoryConfig
    records: list[ObservationRecord] = field(default_factory=list)
    prompt_views: int = 0

    def reset(self) -> None:
        """Start a fresh task when an upstream runner reuses an agent object."""

        self.records.clear()
        self.prompt_views = 0

    def record_observation(
        self,
        message: dict[str, Any],
        *,
        raw_output: str,
        causing_command: str,
        causing_path: str,
    ) -> None:
        content = message.get("content")
        if not isinstance(content, str) or not isinstance(raw_output, str):
            return
        prefix, separator, suffix = content.partition(raw_output)
        if not separator:
            # A custom mini template may transform the output so the boundary
            # cannot be established byte-for-byte.  Failing open is safer than
            # replacing an unknown part of the user message.
            return
        record = ObservationRecord(
            message=message,
            raw_output=raw_output,
            prefix=prefix,
            suffix=suffix,
            causing_command=causing_command,
            causing_path=causing_path,
        )
        self.records.append(record)
        self._write_record_stats(record, status="full", reason="awaiting-followup-action")

    def note_followup(self, signal: PosteriorSignal) -> None:
        """Attach one normal action as posterior evidence to its prior output."""

        for record in reversed(self.records):
            if record.posterior is not None:
                continue
            record.posterior = signal
            record.result = compact_after_followup(
                record.raw_output,
                causing_command=record.causing_command,
                causing_path=record.causing_path,
                posterior=signal,
                config=self.config,
            )
            self._write_record_stats(
                record,
                status=record.result.status,
                reason=record.result.reason,
            )
            return

    def render(self, messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Create the temporary prompt view for one normal agent model call."""

        self.prompt_views += 1
        message_ids = {id(message) for message in messages}
        visible_records = [record for record in self.records if id(record.message) in message_ids]
        cold_records = {
            id(record.message)
            for record in visible_records[
                : max(0, len(visible_records) - self.config.hot_observations)
            ]
            if record.result is not None and record.result.status == "compacted"
        }
        records_by_message = {id(record.message): record for record in visible_records}
        rendered: list[dict[str, Any]] = []
        for message in messages:
            # Model clients and middleware should receive a fully detached
            # prompt object.  A shallow copy would still permit a client that
            # annotates nested ``extra`` or tool metadata to mutate the
            # canonical trajectory accidentally.
            view = copy.deepcopy(dict(message))
            # Keep telemetry in the canonical trajectory only.  OpenAI-style
            # chat APIs should receive only mini's normal message schema.
            view.pop("posterior_history_stats", None)
            record = records_by_message.get(id(message))
            if record is not None and id(message) in cold_records and record.result is not None:
                view["content"] = record.prefix + record.result.text + record.suffix
                saved = record.result.origin_token_cnt - record.result.left_token_cnt
                record.prompt_compaction_count += 1
                record.total_prompt_tokens_saved += saved
                self._write_record_stats(
                    record,
                    status="compacted",
                    reason=record.result.reason,
                )
            rendered.append(view)
        return rendered

    def summary(self) -> dict[str, int]:
        compacted = [
            record
            for record in self.records
            if record.result is not None and record.result.status == "compacted"
        ]
        return {
            "version": 1,
            "model_forward_count": 0,
            "llm_token_count": 0,
            "observations_tracked": len(self.records),
            "eligible_observations": sum(record.posterior is not None for record in self.records),
            "compacted_observations": len(compacted),
            "prompt_views": self.prompt_views,
            "history_prompt_compactions": sum(
                record.prompt_compaction_count for record in compacted
            ),
            "estimated_history_tokens_saved": sum(
                record.total_prompt_tokens_saved for record in compacted
            ),
        }

    def _write_record_stats(self, record: ObservationRecord, *, status: str, reason: str) -> None:
        result = record.result
        record.message["posterior_history_stats"] = {
            "version": 1,
            "method": result.method if result is not None else f"posterior_{self.config.method}",
            "status": status,
            "reason": reason,
            "model_forward_count": 0,
            "llm_token_count": 0,
            "origin_token_cnt": result.origin_token_cnt if result is not None else 0,
            "left_token_cnt": result.left_token_cnt if result is not None else 0,
            "original_line_count": result.original_line_count if result is not None else 0,
            "kept_line_count": result.kept_line_count if result is not None else 0,
            "output_kind": result.output_kind if result is not None else "unknown",
            "posterior_command": record.posterior.command[:500] if record.posterior else "",
            "prompt_compaction_count": record.prompt_compaction_count,
            "total_prompt_tokens_saved": record.total_prompt_tokens_saved,
        }
