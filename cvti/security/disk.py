"""Is the disk encrypted?

A stolen or decommissioned edge PC yields every recorded frame in plaintext
plus all of `events.db`. Full-disk encryption is the accepted v1 answer —
application-level encryption is the follow-up, not a blocker — but "accepted"
only means anything if somebody checked, so the installer checks and the System
panel keeps showing the answer.

Detection is per-platform and best-effort. An honest "unknown" is reported when
it cannot be determined; claiming "encrypted" on a guess would defeat the point.
"""

from __future__ import annotations

import subprocess
import sys

from cvti.logging_setup import get_logger

log = get_logger(__name__)


def _run(cmd: list[str], timeout: float = 6.0) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                             check=False)
        return (out.stdout or "") + (out.stderr or "")
    except (OSError, subprocess.SubprocessError):
        log.debug("disk-encryption probe failed: %s", " ".join(cmd), exc_info=True)
        return ""


def encryption_status() -> dict:
    """{'encrypted': True|False|None, 'mechanism': str, 'detail': str}.

    `None` means undetermined — not encrypted, and not a pass.
    """
    if sys.platform == "darwin":
        out = _run(["fdesetup", "status"])
        if "FileVault is On" in out:
            return {"encrypted": True, "mechanism": "FileVault", "detail": out.strip()[:200]}
        if "FileVault is Off" in out:
            return {"encrypted": False, "mechanism": "FileVault",
                    "detail": "FileVault is off — recorded footage is readable "
                              "from this disk by anyone who takes it"}
        return {"encrypted": None, "mechanism": "FileVault",
                "detail": "could not determine FileVault status"}

    if sys.platform == "win32":
        out = _run(["manage-bde", "-status", "C:"])
        if "Percentage Encrypted: 100" in out or "Fully Encrypted" in out:
            return {"encrypted": True, "mechanism": "BitLocker", "detail": "fully encrypted"}
        if "Fully Decrypted" in out or "Percentage Encrypted: 0" in out:
            return {"encrypted": False, "mechanism": "BitLocker",
                    "detail": "BitLocker is off on the system drive"}
        return {"encrypted": None, "mechanism": "BitLocker",
                "detail": "could not determine BitLocker status "
                          "(manage-bde needs an elevated prompt)"}

    out = _run(["lsblk", "-o", "NAME,TYPE", "-n"])
    if "crypt" in out:
        return {"encrypted": True, "mechanism": "LUKS/dm-crypt",
                "detail": "an encrypted block device is present"}
    if out:
        return {"encrypted": False, "mechanism": "LUKS/dm-crypt",
                "detail": "no encrypted block device found"}
    return {"encrypted": None, "mechanism": "LUKS/dm-crypt",
            "detail": "could not determine disk encryption status"}


def requirement_message(status: dict) -> str:
    """What to tell the person installing this."""
    mech = status.get("mechanism", "full-disk encryption")
    if status.get("encrypted") is True:
        return f"{mech} is on. Recorded footage is protected if this machine is taken."
    if status.get("encrypted") is False:
        return (f"{mech} is OFF. Argus records images of identifiable people; without "
                f"disk encryption, anyone who takes this machine can read all of it. "
                f"Turn {mech} on before putting this site live.")
    return (f"Could not confirm {mech}. Verify it manually before going live — an "
            f"unverified assumption is not a control.")
