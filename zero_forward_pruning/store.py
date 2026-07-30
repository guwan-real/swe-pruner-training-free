from __future__ import annotations

import json
import os
import re
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

RAW_ID_RE = re.compile(r"^[A-Za-z0-9_-]{20,80}$")


@dataclass(frozen=True)
class StoredRaw:
    raw_id: str
    text_path: Path
    metadata_path: Path


class RawStore:
    """Atomic, traversal-safe storage for observations removed from model context."""

    def __init__(
        self,
        root: Path,
        *,
        ttl_hours: float = 72.0,
        max_item_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        if ttl_hours <= 0:
            raise ValueError("ttl_hours must be positive")
        if max_item_bytes < 1:
            raise ValueError("max_item_bytes must be positive")
        self.root = root.expanduser().resolve()
        self.ttl_seconds = ttl_hours * 3600
        self.max_item_bytes = max_item_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    def _validate_id(self, raw_id: str) -> str:
        if not RAW_ID_RE.fullmatch(raw_id):
            raise KeyError("invalid raw observation id")
        return raw_id

    def _paths(self, raw_id: str) -> tuple[Path, Path]:
        raw_id = self._validate_id(raw_id)
        return self.root / f"{raw_id}.txt", self.root / f"{raw_id}.json"

    def save(self, text: str, metadata: Mapping[str, Any] | None = None) -> StoredRaw:
        data = text.encode("utf-8")
        if len(data) > self.max_item_bytes:
            raise ValueError(
                f"raw observation is {len(data)} bytes; maximum is {self.max_item_bytes}"
            )
        raw_id = secrets.token_urlsafe(24)
        text_path, metadata_path = self._paths(raw_id)
        metadata_payload = {
            "raw_id": raw_id,
            "created_at": time.time(),
            "bytes": len(data),
            **dict(metadata or {}),
        }
        text_tmp = self.root / f".{raw_id}.{os.getpid()}.txt.tmp"
        metadata_tmp = self.root / f".{raw_id}.{os.getpid()}.json.tmp"
        try:
            text_tmp.write_bytes(data)
            metadata_tmp.write_text(
                json.dumps(metadata_payload, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(text_tmp, text_path)
            os.replace(metadata_tmp, metadata_path)
        finally:
            text_tmp.unlink(missing_ok=True)
            metadata_tmp.unlink(missing_ok=True)
        return StoredRaw(raw_id=raw_id, text_path=text_path, metadata_path=metadata_path)

    def read(self, raw_id: str) -> str:
        text_path, _ = self._paths(raw_id)
        try:
            return text_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise KeyError("raw observation was not found or has expired") from exc

    def metadata(self, raw_id: str) -> dict[str, Any]:
        _, metadata_path = self._paths(raw_id)
        try:
            value = json.loads(metadata_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise KeyError("raw observation was not found or has expired") from exc
        if not isinstance(value, dict):
            raise ValueError("stored metadata is not an object")
        return value

    def delete(self, raw_id: str) -> None:
        text_path, metadata_path = self._paths(raw_id)
        text_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)

    def purge_expired(self, *, now: float | None = None) -> int:
        cutoff = (time.time() if now is None else now) - self.ttl_seconds
        removed = 0
        for metadata_path in self.root.glob("*.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                raw_id = str(metadata["raw_id"])
                created_at = float(metadata["created_at"])
                self._validate_id(raw_id)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if created_at >= cutoff:
                continue
            self.delete(raw_id)
            removed += 1
        return removed
