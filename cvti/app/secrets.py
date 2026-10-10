"""A small file for the few secrets the app holds on the operator's behalf.

Why a separate file and not site.json: site.json is read by the UI, copied
into support bundles (redacted, but still) and backed up to wherever the
operator pointed backups. An API key for a cloud verifier is worth real
money per month and must live somewhere that is 0600, never bundled, never
on a command line. This is that place. The engine receives the values
through its environment, not its arguments, so they never show in a process
list either.

The file sits beside auth.db in the security directory, which the app
already treats as private.
"""
from __future__ import annotations

import json
import os
import stat
import threading
from pathlib import Path

FILENAME = "secrets.json"


class SecretStore:
    def __init__(self, directory: str | Path) -> None:
        self.path = Path(directory) / FILENAME
        self._lock = threading.Lock()

    def _read(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        # Create 0600 before any byte of secret lands in it.
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        try:
            os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
        tmp.replace(self.path)

    def get(self, name: str) -> str:
        with self._lock:
            value = self._read().get(name)
        return str(value) if isinstance(value, str) else ""

    def has(self, name: str) -> bool:
        return bool(self.get(name))

    def set(self, name: str, value: str) -> None:
        with self._lock:
            data = self._read()
            data[name] = str(value)
            self._write(data)

    def delete(self, name: str) -> None:
        with self._lock:
            data = self._read()
            if name in data:
                del data[name]
                self._write(data)


VERIFIER_KEY = "verifier_api_key"
