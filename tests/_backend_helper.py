"""Signed-in backends for tests.

Every consequential ConsoleBackend method is now behind a permission check, so
a test that configures cameras or reviews an alert has to be somebody. This
keeps that one line rather than seven copies of it, and makes the role a test
runs as an explicit choice — which is what the role tests then vary.
"""
from __future__ import annotations

from cvti.app.console_backend import ConsoleBackend

OWNER_PASSWORD = "a-strong-test-password"


def signed_in(role: str = "owner", **kwargs) -> ConsoleBackend:
    """A backend with an owner account, signed in as `role`."""
    backend = ConsoleBackend(**kwargs)
    backend.create_first_owner("owner", OWNER_PASSWORD)
    if role != "owner":
        backend.add_user(role, OWNER_PASSWORD, role=role)
        backend.sign_out()
        backend.sign_in(role, OWNER_PASSWORD)
    return backend
