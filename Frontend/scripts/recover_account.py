"""Terminal-only recovery for an OS user who owns the local account database.

Not an IPC endpoint. Never accepts passwords as command-line arguments.
"""
from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cvti.security.accounts import AccountStore
from cvti.security.audit import AuditLog


def auth_path(events_db):
    path = Path(events_db).resolve().parent / 'auth.db'
    if not path.is_file():
        raise ValueError('No existing account database at ' + str(path))
    if hasattr(os, 'getuid') and path.stat().st_uid != os.getuid():
        raise PermissionError('Run recovery as the OS user who owns this database.')
    if not os.access(path, os.R_OK | os.W_OK):
        raise PermissionError('Recovery requires local read/write access to the account database.')
    return path


def reset_account(events_db, username, password, *, actor):
    path = auth_path(events_db)
    if len(password) < 12:
        raise ValueError('Use at least 12 characters for the replacement password.')
    store = AccountStore(path)
    audit = None
    try:
        if store.user(username) is None:
            raise ValueError('Account not found; no accounts were changed.')
        audit = AuditLog(path.parent / 'audit.db')
        audit.record(actor, 'role_change', 'user:' + username,
                     {'local_password_recovery': 'started'})
        store.set_password(username, password)
        with sqlite3.connect(path) as connection:
            connection.execute('DELETE FROM login_failures WHERE username=?', (username,))
        audit.record(actor, 'role_change', 'user:' + username,
                     {'local_password_recovery': 'completed', 'sessions_revoked': True})
    finally:
        store.close()
        if audit is not None:
            audit.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', required=True, help='Existing events.db path; auth.db is beside it')
    args = parser.parse_args()
    if not sys.stdin.isatty():
        parser.error('Run interactively in Terminal. Passwords must not be piped or passed as arguments.')
    try:
        path = auth_path(args.db)
        store = AccountStore(path)
        try:
            users = store.list_users()
        finally:
            store.close()
        print('Account database:', path)
        print('Existing accounts:')
        for user in users:
            print('  ' + user.username + ' (' + user.role + ')')
        username = input('Account to recover: ').strip()
        if username not in [user.username for user in users]:
            raise ValueError('Choose an existing account. Nothing changed.')
        print('Only this password and its active sessions will change. Evidence and other accounts are retained.')
        if input('Type RESET to continue: ').strip() != 'RESET':
            print('Cancelled. Nothing changed.')
            return 0
        password = getpass.getpass('New password (12+ characters): ')
        if password != getpass.getpass('Confirm new password: '):
            raise ValueError('Passwords do not match. Nothing changed.')
        reset_account(args.db, username, password, actor='local-os:' + getpass.getuser())
        print('Password reset. Sign in with the selected account and new password.')
        return 0
    except (ValueError, PermissionError, OSError, sqlite3.Error) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
