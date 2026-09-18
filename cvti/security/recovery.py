"""Terminal-only recovery for the OS user who owns the local account database.

Not an IPC endpoint, and never a password on a command line. Lives here, in
the package, rather than in a loose script because an INSTALLED customer has
no repo and no Python: the frozen `argus-api` binary exposes this same flow
through `--recover-account`, and Frontend/scripts/recover_account.py is a
thin wrapper over it for development. One implementation, so the guards
cannot drift apart between the two ways in.
"""
from __future__ import annotations

import getpass
import os
import sqlite3
import sys
from pathlib import Path

MIN_PASSWORD = 12


def auth_path(events_db: str | Path) -> Path:
    """The account database beside an events store, if this user may fix it."""
    path = Path(events_db).resolve().parent / "auth.db"
    if not path.is_file():
        raise ValueError("No existing account database at " + str(path))
    if hasattr(os, "getuid") and path.stat().st_uid != os.getuid():
        raise PermissionError("Run recovery as the OS user who owns this database.")
    if not os.access(path, os.R_OK | os.W_OK):
        raise PermissionError(
            "Recovery requires local read/write access to the account database.")
    return path


def reset_account(events_db: str | Path, username: str, password: str, *, actor: str) -> None:
    """Replace one account's password and revoke its sessions. Nothing else."""
    from cvti.security.accounts import AccountStore
    from cvti.security.audit import AuditLog

    path = auth_path(events_db)
    if len(password) < MIN_PASSWORD:
        raise ValueError(
            f"Use at least {MIN_PASSWORD} characters for the replacement password.")
    store = AccountStore(path)
    audit = None
    try:
        if store.user(username) is None:
            raise ValueError("Account not found; no accounts were changed.")
        audit = AuditLog(path.parent / "audit.db")
        audit.record(actor, "role_change", "user:" + username,
                     {"local_password_recovery": "started"})
        store.set_password(username, password)
        with sqlite3.connect(path) as connection:
            connection.execute("DELETE FROM login_failures WHERE username=?", (username,))
        audit.record(actor, "role_change", "user:" + username,
                     {"local_password_recovery": "completed", "sessions_revoked": True})
    finally:
        store.close()
        if audit is not None:
            audit.close()


def run_interactive(events_db: str | Path) -> int:
    """The operator-facing flow: list accounts, confirm, prompt, reset."""
    from cvti.security.accounts import AccountStore

    if not sys.stdin.isatty():
        print("Run interactively in Terminal. Passwords must not be piped or "
              "passed as arguments.", file=sys.stderr)
        return 1
    try:
        path = auth_path(events_db)
        store = AccountStore(path)
        try:
            users = store.list_users()
        finally:
            store.close()
        print("Account database:", path)
        print("Existing accounts:")
        for user in users:
            print("  " + user.username + " (" + user.role + ")")
        username = input("Account to recover: ").strip()
        if username not in [user.username for user in users]:
            raise ValueError("Choose an existing account. Nothing changed.")
        print("Only this password and its active sessions will change. "
              "Evidence and other accounts are retained.")
        if input("Type RESET to continue: ").strip() != "RESET":
            print("Cancelled. Nothing changed.")
            return 0
        password = getpass.getpass(f"New password ({MIN_PASSWORD}+ characters): ")
        if password != getpass.getpass("Confirm new password: "):
            raise ValueError("Passwords do not match. Nothing changed.")
        reset_account(events_db, username, password,
                      actor="local-os:" + getpass.getuser())
        print("Password reset. Sign in with the selected account and new password.")
        return 0
    except (ValueError, PermissionError, OSError, sqlite3.Error) as error:
        print(str(error), file=sys.stderr)
        return 1
