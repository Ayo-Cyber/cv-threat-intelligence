"""Local accounts, password hashing, and sessions.

Today anyone with physical or network access has complete control: every
camera, every recording, every rule, every detector. The failure that matters
most is not misuse of the cameras — it is that **anyone can silently disable
detection**, and for a security product that is a contradiction in terms.

Hashing avoids argon2 and bcrypt, which are third-party wheels that would have
to build inside a PyInstaller bundle on three platforms. It prefers
`hashlib.scrypt` (memory-hard, OWASP-recommended) and falls back to
PBKDF2-HMAC-SHA256 at OWASP's iteration count.

The fallback is not theoretical. `hashlib.scrypt` needs OpenSSL 1.1+, and the
Python this was developed against is built on LibreSSL 2.8.3, where the
attribute simply does not exist — so a stdlib-only choice still has to handle a
stdlib function being absent. Each hash records the KDF that produced it, so a
machine with scrypt and a machine without can read the same database, and the
parameters can be raised later without invalidating existing accounts.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from cvti.logging_setup import get_logger

log = get_logger(__name__)

SCRYPT_N = 2 ** 15          # 32 MiB — a desktop-friendly point on the curve
SCRYPT_R = 8
SCRYPT_P = 1
# OWASP's floor for PBKDF2-HMAC-SHA256. ~0.15s here: slow enough to matter to an
# offline attacker, fast enough that nobody disables the login over it.
PBKDF2_ROUNDS = 600_000
SALT_BYTES = 16
KEY_BYTES = 32

HAVE_SCRYPT = hasattr(hashlib, "scrypt")

# Lockout: slow an online guesser to a crawl without letting anyone lock a real
# operator out of a live security system indefinitely.
MAX_FAILED = 5
LOCKOUT_SECONDS = 300.0

DEFAULT_SESSION_TIMEOUT = 8 * 3600.0     # a shift
ROLES = ("owner", "operator", "installer")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    salt BLOB NOT NULL,
    hash BLOB NOT NULL,
    kdf TEXT NOT NULL,            -- parameters, so they can be raised later
    must_change INTEGER DEFAULT 0,
    created_at REAL,
    last_login REAL
);
CREATE TABLE IF NOT EXISTS login_failures (
    username TEXT, at REAL
);
CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    username TEXT NOT NULL,
    created_at REAL,
    expires_at REAL
);
"""


@dataclass
class User:
    username: str
    role: str
    must_change: bool = False
    last_login: float = 0.0


class AuthError(Exception):
    """Authentication refused. Deliberately says little; the log says more."""


class UnsupportedKDF(Exception):
    """A hash this interpreter cannot recompute — not a wrong password."""


def hash_password(password: str, salt: bytes | None = None, kdf: str = "") -> tuple:
    """Hash a password. Returns (salt, digest, kdf-descriptor).

    `kdf` re-derives with the parameters of an existing hash; empty picks the
    best this interpreter can actually do.
    """
    salt = salt or os.urandom(SALT_BYTES)
    pw = password.encode("utf-8")

    if kdf.startswith("scrypt$"):
        _, n, r, p = kdf.split("$")
        n, r, p = int(n), int(r), int(p)
        if not HAVE_SCRYPT:
            # A database written on a machine with scrypt, opened on one
            # without. Say so plainly rather than failing every login as if the
            # passwords were wrong.
            raise UnsupportedKDF(
                "this account was hashed with scrypt, which this Python build "
                "does not provide (needs OpenSSL 1.1+); reset the password on a "
                "build that does, or recreate the account")
        return salt, hashlib.scrypt(pw, salt=salt, n=n, r=r, p=p, dklen=KEY_BYTES,
                                    maxmem=n * r * 256), kdf

    if kdf.startswith("pbkdf2$"):
        rounds = int(kdf.split("$")[2])
        return salt, hashlib.pbkdf2_hmac("sha256", pw, salt, rounds, KEY_BYTES), kdf

    if kdf:
        raise UnsupportedKDF(f"unknown password hash format {kdf!r}")

    if HAVE_SCRYPT:
        digest = hashlib.scrypt(pw, salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P,
                                dklen=KEY_BYTES, maxmem=SCRYPT_N * SCRYPT_R * 256)
        return salt, digest, f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}"
    digest = hashlib.pbkdf2_hmac("sha256", pw, salt, PBKDF2_ROUNDS, KEY_BYTES)
    return salt, digest, f"pbkdf2$sha256${PBKDF2_ROUNDS}"


def _verify_password(password: str, salt: bytes, expected: bytes, kdf: str) -> bool:
    try:
        _, digest, _ = hash_password(password, salt, kdf)
    except UnsupportedKDF:
        log.error("cannot verify password: %s", kdf, exc_info=True)
        return False
    # Constant-time: a timing difference here leaks how much of the hash matched.
    return hmac.compare_digest(digest, expected)


class AccountStore:
    """Users and sessions, in their own database.

    Separate from `events.db` on purpose: evidence and credentials have
    different lifecycles, different backup rules, and different blast radius if
    one is copied off the machine.
    """

    def __init__(self, db_path: str | Path, session_timeout: float = DEFAULT_SESSION_TIMEOUT):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_timeout = session_timeout
        self._db = sqlite3.connect(self.db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)
        self._db.commit()
        try:      # credentials should not be world-readable
            os.chmod(self.db_path, 0o600)
        except OSError:
            log.debug("could not tighten permissions on %s", self.db_path, exc_info=True)

    # --- users -----------------------------------------------------------
    def create_user(self, username: str, password: str, role: str = "operator",
                    must_change: bool = False) -> User:
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")
        if not username or not username.strip():
            raise ValueError("username is required")
        if len(password) < 8:
            raise ValueError("password must be at least 8 characters")
        salt, digest, kdf = hash_password(password)
        try:
            self._db.execute(
                "INSERT INTO users (username, role, salt, hash, kdf, must_change, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (username, role, salt, digest, kdf, int(must_change), time.time()))
            self._db.commit()
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"user {username!r} already exists") from exc
        log.info("account created: %s (%s)", username, role)
        return User(username, role, must_change)

    def user(self, username: str) -> User | None:
        row = self._db.execute(
            "SELECT username, role, must_change, last_login FROM users WHERE username=?",
            (username,)).fetchone()
        return User(row["username"], row["role"], bool(row["must_change"]),
                    row["last_login"] or 0.0) if row else None

    def list_users(self) -> list[User]:
        return [User(r["username"], r["role"], bool(r["must_change"]), r["last_login"] or 0.0)
                for r in self._db.execute(
                    "SELECT username, role, must_change, last_login FROM users ORDER BY username")]

    def any_users(self) -> bool:
        return bool(self._db.execute("SELECT 1 FROM users LIMIT 1").fetchone())

    def set_password(self, username: str, password: str) -> None:
        if len(password) < 8:
            raise ValueError("password must be at least 8 characters")
        salt, digest, kdf = hash_password(password)
        self._db.execute(
            "UPDATE users SET salt=?, hash=?, kdf=?, must_change=0 WHERE username=?",
            (salt, digest, kdf, username))
        self._db.commit()
        # Any session opened with the old credential is no longer trustworthy.
        self._db.execute("DELETE FROM sessions WHERE username=?", (username,))
        self._db.commit()
        log.info("password changed for %s; existing sessions revoked", username)

    def set_role(self, username: str, role: str) -> None:
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}")
        self._db.execute("UPDATE users SET role=? WHERE username=?", (role, username))
        self._db.commit()
        log.info("role for %s set to %s", username, role)

    def delete_user(self, username: str) -> None:
        self._db.execute("DELETE FROM users WHERE username=?", (username,))
        self._db.execute("DELETE FROM sessions WHERE username=?", (username,))
        self._db.commit()

    # --- authentication --------------------------------------------------
    def _recent_failures(self, username: str) -> int:
        cutoff = time.time() - LOCKOUT_SECONDS
        self._db.execute("DELETE FROM login_failures WHERE at < ?", (cutoff,))
        self._db.commit()
        return self._db.execute(
            "SELECT COUNT(*) FROM login_failures WHERE username=? AND at >= ?",
            (username, cutoff)).fetchone()[0]

    def locked_out(self, username: str) -> bool:
        return self._recent_failures(username) >= MAX_FAILED

    def authenticate(self, username: str, password: str) -> User:
        """Return the user, or raise AuthError. Never says which half was wrong."""
        if self.locked_out(username):
            log.warning("login refused for %s: locked out after %d failures",
                        username, MAX_FAILED)
            raise AuthError("too many failed attempts; try again later")

        row = self._db.execute(
            "SELECT username, role, salt, hash, kdf, must_change FROM users WHERE username=?",
            (username,)).fetchone()
        if row is None:
            # Hash anyway so a missing user and a wrong password take the same
            # time — otherwise this endpoint enumerates valid usernames.
            hash_password(password)
            self._record_failure(username)
            raise AuthError("invalid username or password")

        if not _verify_password(password, row["salt"], row["hash"], row["kdf"]):
            self._record_failure(username)
            log.warning("failed login for %s", username)
            raise AuthError("invalid username or password")

        self._db.execute("DELETE FROM login_failures WHERE username=?", (username,))
        self._db.execute("UPDATE users SET last_login=? WHERE username=?",
                         (time.time(), username))
        self._db.commit()
        log.info("login: %s (%s)", username, row["role"])
        return User(row["username"], row["role"], bool(row["must_change"]))

    def _record_failure(self, username: str) -> None:
        self._db.execute("INSERT INTO login_failures (username, at) VALUES (?,?)",
                         (username, time.time()))
        self._db.commit()

    # --- sessions --------------------------------------------------------
    def open_session(self, username: str, timeout: float | None = None) -> str:
        token = secrets.token_urlsafe(32)
        now = time.time()
        self._db.execute(
            "INSERT INTO sessions (token, username, created_at, expires_at) VALUES (?,?,?,?)",
            (token, username, now, now + (timeout or self.session_timeout)))
        self._db.commit()
        return token

    def session_user(self, token: str) -> User | None:
        """The user behind a token, or None if absent or expired."""
        if not token:
            return None
        self._db.execute("DELETE FROM sessions WHERE expires_at < ?", (time.time(),))
        self._db.commit()
        row = self._db.execute(
            "SELECT u.username, u.role, u.must_change, u.last_login FROM sessions s "
            "JOIN users u ON u.username = s.username "
            "WHERE s.token = ? AND s.expires_at >= ?", (token, time.time())).fetchone()
        return User(row["username"], row["role"], bool(row["must_change"]),
                    row["last_login"] or 0.0) if row else None

    def close_session(self, token: str) -> None:
        self._db.execute("DELETE FROM sessions WHERE token=?", (token,))
        self._db.commit()

    def close(self) -> None:
        self._db.close()


def bootstrap_password() -> str:
    """A random first-run credential. Never a fixed default.

    A shipped default password is a published password: it ends up in a forum
    post, and every site that never changed it is open. This is generated per
    install, shown once, and must be changed before anything else works.
    """
    return secrets.token_urlsafe(12)
