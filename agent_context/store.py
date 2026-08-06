from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from agent_context.models import Observation, ObservationRuntime


class ObservationNotFound(KeyError):
    pass


class ObservationStore(Protocol):
    def add(self, runtime: ObservationRuntime) -> None: ...

    def get(self, observation_id: str) -> ObservationRuntime: ...

    def values(self) -> tuple[ObservationRuntime, ...]: ...

    def clear(self) -> None: ...


class InMemoryObservationStore:
    """In-process canonical store with immutable observation payloads."""

    def __init__(self) -> None:
        self._records: dict[str, ObservationRuntime] = {}
        self._order: list[str] = []

    def add(self, runtime: ObservationRuntime) -> None:
        observation_id = runtime.observation.id
        if observation_id in self._records:
            raise ValueError(f"duplicate observation id: {observation_id}")
        self._records[observation_id] = runtime
        self._order.append(observation_id)

    def get(self, observation_id: str) -> ObservationRuntime:
        try:
            return self._records[observation_id]
        except KeyError as exc:
            raise ObservationNotFound(observation_id) from exc

    def values(self) -> tuple[ObservationRuntime, ...]:
        return tuple(self._records[value] for value in self._order)

    def observations(self) -> tuple[Observation, ...]:
        return tuple(runtime.observation for runtime in self.values())

    def __iter__(self) -> Iterable[ObservationRuntime]:
        return iter(self.values())

    def __len__(self) -> int:
        return len(self._records)

    def clear(self) -> None:
        self._records.clear()
        self._order.clear()
