"""No tracked config may carry a live credential.

The notify spec embeds real secrets: a Telegram bot token
(telegram:<bot_id>:<token>:<chat_id>) and a Twilio auth token
(whatsapp:<sid>:<token>:<from>:<to>). On an installed machine those live in
the per-user data directory, which is nobody's repository. In the checkout
they land in configs/site_live.json -- which IS tracked -- the moment anyone
points a dev run at their own phone.

On 20 Sep two tracked files held a live bot token in the working tree. The
history was clean, so nothing had leaked yet; one `git add -A` would have
changed that. Remembering not to do it is not a control. This is.

Untracked files are skipped deliberately: they never ship and never reach a
commit, and failing on them would make the suite unrunnable for anyone
testing against their own phone.
"""
from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Shapes that are real secrets, not placeholders.
SECRETS = (
    # telegram:<digits>:<token>:<chat> -- a bot id is numeric and the token long
    re.compile(r"telegram:\d{6,}:[A-Za-z0-9_-]{20,}"),
    # whatsapp:AC<sid>:<token>:...
    re.compile(r"whatsapp:AC[0-9a-zA-Z]{10,}:[0-9a-zA-Z]{10,}"),
    re.compile(r"TWILIO_AUTH_TOKEN\s*[:=]\s*['\"][0-9a-zA-Z]{20,}"),
)


def tracked_configs() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "configs"],
            cwd=ROOT, capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            raise OSError(out.stderr[:120])
        names = [n for n in out.stdout.split("\n") if n.strip().endswith(".json")]
        return [ROOT / n for n in names if (ROOT / n).is_file()]
    except (OSError, subprocess.SubprocessError):
        # No git (a source tarball): scan what is here, which is tracked anyway.
        return sorted((ROOT / "configs").rglob("*.json"))


class TrackedConfigsCarryNoSecrets(unittest.TestCase):
    def test_no_tracked_config_embeds_a_credential(self):
        offenders = []
        for path in tracked_configs():
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            for pattern in SECRETS:
                if pattern.search(text):
                    offenders.append(path.relative_to(ROOT).as_posix())
                    break
        self.assertEqual(
            offenders, [],
            "these TRACKED config files contain a live credential, and a single "
            "`git add -A` would publish it:\n  " + "\n  ".join(offenders) +
            "\n\nMove the notify spec out of the tracked file: point the dev run "
            "at a site config under runs/ or your scratch directory instead.")

    def test_the_patterns_actually_match_a_real_spec(self):
        """A guard that cannot recognise the thing it guards is theatre."""
        samples = [
            "telegram:8691681982:AAH216L_pYnTx-L576-PF2JKvwLtUfKXGYc:1883642843",
            "whatsapp:ACa1b2c3d4e5f6:0123456789abcdef0123:+14155238886:+2348012345678",
        ]
        for sample in samples:
            with self.subTest(sample=sample.split(":")[0]):
                self.assertTrue(any(p.search(sample) for p in SECRETS),
                                f"the guard does not recognise {sample.split(':')[0]}")

    def test_placeholders_are_not_flagged(self):
        """Docs and examples must stay writable."""
        harmless = [
            '"notify": "console"',
            '"notify": "telegram:<token>:<chat_id>"',
            '"notify": "webhook:https://example.com/hook"',
            "telegram:123:AAH:456",          # the short fake used in unit tests
        ]
        for sample in harmless:
            with self.subTest(sample=sample):
                self.assertFalse(any(p.search(sample) for p in SECRETS),
                                 f"false positive on {sample!r}")


if __name__ == "__main__":
    unittest.main()
