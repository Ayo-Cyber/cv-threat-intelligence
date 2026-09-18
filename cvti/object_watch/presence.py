"""Stable, bounded presence admission for object-watch matches."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Hashable


@dataclass(frozen=True)
class PresenceToken:
    key: tuple[Hashable, ...]
    encounter: int


@dataclass
class _Presence:
    encounter: int = 1
    observations: deque[Hashable] = field(default_factory=deque)
    last_observed_at: float = 0.0
    committed: bool = False

class PresenceDebouncer:
    """Require distinct successful samples and emit once per encounter.

    ``ready`` does not consume eligibility.  The caller must call ``commit``
    only after it has admitted the associated result to its bounded queue.
    """

    def __init__(self, *, min_observations: int = 2, rearm_gap_seconds: float = 2.0,
                 max_entries: int = 512) -> None:
        if isinstance(min_observations, bool) or min_observations < 1:
            raise ValueError("min_observations must be positive")
        if rearm_gap_seconds <= 0 or max_entries <= 0:
            raise ValueError("presence bounds must be positive")
        self.min_observations = int(min_observations)
        self.rearm_gap_seconds = float(rearm_gap_seconds)
        self.max_entries = int(max_entries)
        self._states: dict[tuple[Hashable, ...], _Presence] = {}

    def ready(self, camera_id: Hashable, source_generation: Hashable,
              rule_id: Hashable, target_id: Hashable, track_id: Hashable,
              observation_id: Hashable, observed_at: float, *,
              available: bool = True, track_ended: bool = False) -> PresenceToken | None:
        key = (camera_id, source_generation, rule_id, target_id, track_id)
        state = self._states.get(key)
        if state is None:
            self._trim()
            state = self._states.setdefault(key, _Presence(last_observed_at=float(observed_at)))
        if not available:
            return None
        now = float(observed_at)
        if track_ended or (state.observations and now - state.last_observed_at > self.rearm_gap_seconds):
            state.encounter += 1
            state.observations.clear()
            state.committed = False
        state.last_observed_at = now
        if observation_id not in state.observations:
            state.observations.append(observation_id)
            while len(state.observations) > self.min_observations:
                state.observations.popleft()
        if state.committed or len(state.observations) < self.min_observations:
            return None
        return PresenceToken(key, state.encounter)

    def commit(self, token: PresenceToken) -> bool:
        state = self._states.get(token.key)
        if state is None or state.encounter != token.encounter or state.committed:
            return False
        state.committed = True
        return True

    def reset_camera(self, camera_id: Hashable, source_generation: Hashable | None = None) -> None:
        self._states = {
            key: value for key, value in self._states.items()
            if not (key[0] == camera_id and (source_generation is None or key[1] != source_generation))
        }

    def _trim(self) -> None:
        if len(self._states) < self.max_entries:
            return
        victim = min(self._states, key=lambda key: self._states[key].last_observed_at)
        del self._states[victim]
