"""Terminal-only recovery for an OS user who owns the local account database.

Not an IPC endpoint. Never accepts passwords as command-line arguments.

The flow itself lives in cvti.security.recovery so the INSTALLED app can
offer the same recovery through `argus-api --recover-account`: a customer
with no repo and no Python cannot run this script, and before that binary
flag existed a locked-out installation had no way back in at all. Importing
the guards rather than repeating them keeps the two entrances identical.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cvti.security.recovery import auth_path, reset_account, run_interactive  # noqa: E402,F401


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', required=True,
                        help='Existing events.db path; auth.db is beside it')
    args = parser.parse_args()
    return run_interactive(args.db)


if __name__ == '__main__':
    raise SystemExit(main())
