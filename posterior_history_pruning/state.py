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

OUTPUT_OPEN = "<output>"
OUTPUT_CLOSE = "</output>"
OUTPUT_HEAD_OPEN = "<output_head>"
OUTPUT_HEAD_CLOSE = "</output_head>"
OUTPUT_TAIL_OPEN = "<output_tail>"
OUTPUT_TAIL_CLOSE = "</output_tail>"
OFFICIAL_TRUNCATED_CHUNK_CHARS = 5000


@dataclass(frozen=True)
class ObservationBoundary:
    """A selector input and the exact prompt replacement boundary."""

    selection_output: str
    prefix: str
    suffix: str
    mode: str


@dataclass
class ObservationRecord:
    """A canonical observation plus a separately rendered prompt view."""

    message: dict[str, Any]
    selection_output: str
    prefix: str
    suffix: str
    causing_command: str
    causing_path: str
    boundary_mode: str
    source_output_chars: int
    posterior: PosteriorSignal | None = None
    result: CompactionResult | None = None
    prompt_compaction_count: int = 0
    total_prompt_tokens_saved: int = 0


def _direct_boundary(content: str, raw_output: str) -> ObservationBoundary | None:
    position = content.find(raw_output)
    if position < 0:
        return None
    end = position + len(raw_output)
    return ObservationBoundary(
        selection_output=raw_output,
        prefix=content[:position],
        suffix=content[end:],
        mode="verbatim",
    )


def _tagged_output_boundary(content: str, raw_output: str) -> ObservationBoundary | None:
    """Locate a complete output inside the official ``<output>`` wrapper."""

    output_open = content.find(OUTPUT_OPEN)
    output_close = content.rfind(OUTPUT_CLOSE)
    if output_open < 0 or output_close <= output_open:
        return None
    payload_start = output_open + len(OUTPUT_OPEN)
    position = content.find(raw_output, payload_start, output_close)
    if position < 0:
        return None
    end = position + len(raw_output)
    return ObservationBoundary(
        selection_output=raw_output,
        prefix=content[:position],
        suffix=content[end:],
        mode="official-output",
    )


def _tagged_truncated_boundary(content: str, raw_output: str) -> ObservationBoundary | None:
    """Recognize the official mini head/tail rendering conservatively.

    The official SWE-Bench template renders the first and last 5,000
    characters under distinct tags once an output reaches 10,000 characters.
    The complete environment output is therefore not a substring of the
    message. Both raw anchors must match before the template is accepted.
    """

    if len(raw_output) < OFFICIAL_TRUNCATED_CHUNK_CHARS * 2:
        return None
    head_open = content.find(OUTPUT_HEAD_OPEN)
    tail_open = content.rfind(OUTPUT_TAIL_OPEN)
    tail_close = content.rfind(OUTPUT_TAIL_CLOSE)
    if head_open < 0 or tail_open < 0 or tail_close < 0:
        return None
    head_close = content.rfind(OUTPUT_HEAD_CLOSE, head_open, tail_open)
    if not (head_open < head_close < tail_open < tail_close):
        return None

    head_payload_start = head_open + len(OUTPUT_HEAD_OPEN)
    tail_payload_start = tail_open + len(OUTPUT_TAIL_OPEN)
    head_payload = content[head_payload_start:head_close]
    tail_payload = content[tail_payload_start:tail_close]
    raw_head = raw_output[:OFFICIAL_TRUNCATED_CHUNK_CHARS]
    raw_tail = raw_output[-OFFICIAL_TRUNCATED_CHUNK_CHARS:]
    if raw_head not in head_payload or raw_tail not in tail_payload:
        return None

    omitted_chars = max(0, len(raw_output) - len(raw_head) - len(raw_tail))
    selection_output = (
        raw_head
        + f"\n... [mini-swe-agent already elided {omitted_chars} characters] ...\n"
        + raw_tail
    )
    replacement_open = '<output_posterior_view source="official-head-tail">\n'
    replacement_close = "\n</output_posterior_view>"
    return ObservationBoundary(
        selection_output=selection_output,
        prefix=content[:head_open] + replacement_open,
        suffix=replacement_close + content[tail_close + len(OUTPUT_TAIL_CLOSE) :],
        mode="official-head-tail",
    )


def locate_observation_boundary(content: str, raw_output: str) -> ObservationBoundary | None:
    """Locate exactly what the model saw using the strictest contract first."""

    if not raw_output:
        return None
    tagged = _tagged_output_boundary(content, raw_output)
    if tagged is not None:
        return tagged
    if OUTPUT_OPEN in content or OUTPUT_CLOSE in content:
        # A partially matching official wrapper must not fall back to a global
        # substring that could accidentally target <returncode> or a warning.
        return None
    direct = _direct_boundary(content, raw_output)
    if direct is not None:
        return direct
    return _tagged_truncated_boundary(content, raw_output)


@dataclass
class PosteriorHistoryState:
    """Per-agent state that never changes canonical observation content.

    ``agent.messages`` remains the full, auditable trajectory.  ``render``
    returns a copied message list only for the next model request.
    """

    config: PosteriorHistoryConfig
    records: list[ObservationRecord] = field(default_factory=list)
    prompt_views: int = 0
    observations_seen: int = 0
    untracked_observations: int = 0

    def reset(self) -> None:
        """Start a fresh task when an upstream runner reuses an agent object."""

        self.records.clear()
        self.prompt_views = 0
        self.observations_seen = 0
        self.untracked_observations = 0

    def record_observation(
        self,
        message: dict[str, Any],
        *,
        raw_output: str,
        causing_command: str,
        causing_path: str,
    ) -> bool:
        if raw_output == "":
            # Empty command output has no history payload to compact and is
            # not a boundary failure.
            return False
        self.observations_seen += 1
        content = message.get("content")
        if not isinstance(content, str) or not isinstance(raw_output, str):
            self._write_untracked_stats(
                message,
                reason="non-text-observation",
                raw_output_chars=len(raw_output) if isinstance(raw_output, str) else 0,
            )
            return False
        boundary = locate_observation_boundary(content, raw_output)
        if boundary is None:
            # A custom mini template may transform the output so the boundary
            # cannot be established safely. Keep the prompt unchanged, but
            # emit telemetry so an entire class of misses is not invisible.
            self._write_untracked_stats(
                message,
                reason="rendered-output-boundary-not-found",
                raw_output_chars=len(raw_output),
            )
            return False
        record = ObservationRecord(
            message=message,
            selection_output=boundary.selection_output,
            prefix=boundary.prefix,
            suffix=boundary.suffix,
            causing_command=causing_command,
            causing_path=causing_path,
            boundary_mode=boundary.mode,
            source_output_chars=len(raw_output),
        )
        self.records.append(record)
        self._write_record_stats(record, status="full", reason="awaiting-followup-action")
        return True

    def note_followup(self, signal: PosteriorSignal) -> None:
        """Attach one normal action as posterior evidence to its prior output."""

        for record in reversed(self.records):
            if record.posterior is not None:
                continue
            record.posterior = signal
            record.result = compact_after_followup(
                record.selection_output,
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
            "observations_seen": self.observations_seen,
            "observations_tracked": len(self.records),
            "untracked_observations": self.untracked_observations,
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
            "boundary_mode": record.boundary_mode,
            "source_output_chars": record.source_output_chars,
            "visible_output_chars": len(record.selection_output),
            "posterior_command": record.posterior.command[:500] if record.posterior else "",
            "prompt_compaction_count": record.prompt_compaction_count,
            "total_prompt_tokens_saved": record.total_prompt_tokens_saved,
        }

    def _write_untracked_stats(
        self,
        message: dict[str, Any],
        *,
        reason: str,
        raw_output_chars: int,
    ) -> None:
        self.untracked_observations += 1
        message["posterior_history_stats"] = {
            "version": 1,
            "method": f"posterior_{self.config.method}",
            "status": "untracked",
            "reason": reason,
            "model_forward_count": 0,
            "llm_token_count": 0,
            "origin_token_cnt": 0,
            "left_token_cnt": 0,
            "original_line_count": 0,
            "kept_line_count": 0,
            "output_kind": "unknown",
            "boundary_mode": "unrecognized",
            "source_output_chars": raw_output_chars,
            "visible_output_chars": 0,
            "posterior_command": "",
            "prompt_compaction_count": 0,
            "total_prompt_tokens_saved": 0,
        }
