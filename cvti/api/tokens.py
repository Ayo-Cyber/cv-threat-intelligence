"""Bearer-token store for the API.

The console keeps one session; an API serves many clients at once, so tokens
live here, mapped to {username, role, expiry}. In-memory on purpose: a control
plane restart re-authenticates its clients, which is the safe default for a
security product. Swappable for a persistent store later without touching call
sites.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass


DEFAULT_TTL_SECONDS = 12 * 3600


@dataclass
class Principal:
    username: str
    role: str
    expires_at: float

    @property
    def expired(self) -> bool:
        return time.time() >= self.expires_at


class TokenStore:
    def __init__(self, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._by_token: dict[str, Principal] = {}
        self._lock = threading.Lock()

    def mint(self, username: str, role: str) -> tuple[str, Principal]:
        token = secrets.token_urlsafe(32)
        principal = Principal(username=username, role=role,
                              expires_at=time.time() + self._ttl)
        with self._lock:
            self._by_token[token] = principal
        return token, principal

    def resolve(self, token: str | None) -> Principal | None:
        if not token:
            return None
        with self._lock:
            principal = self._by_token.get(token)
            if principal is None:
                return None
            if principal.expired:
                self._by_token.pop(token, None)
                return None
            return principal

    def revoke(self, token: str | None) -> None:
        if not token:
            return
        with self._lock:
            self._by_token.pop(token, None)
