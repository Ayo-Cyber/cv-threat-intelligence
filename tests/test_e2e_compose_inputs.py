"""The E2E stack must only depend on files that are IN the repository.

27 Aug: the container E2E passed on the author's laptop for two days and
failed every run in CI, because the camera container published
`data/test_clips/crowd_01.mp4` — a file matched by `data/test_clips/*` in
.gitignore. It existed locally and nowhere else. CI decoded zero frames, the
engine sat in its reconnect loop for fifteen minutes, and the failure surfaced
as `DESCRIBE failed: 404` from the RTSP server: a symptom three layers away
from the cause.

A green test on a machine that has extra files is not a green test. These
assertions read the compose file and check that every input it names is
tracked by git, so the next person to swap a clip finds out here instead of
in a thirty-minute CI run.
"""
from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "tests" / "e2e" / "docker-compose.yml"


def _tracked(rel: str) -> bool:
    r = subprocess.run(["git", "ls-files", "--error-unmatch", rel],
                       cwd=ROOT, capture_output=True)
    return r.returncode == 0


class E2EComposeInputsTest(unittest.TestCase):

    def setUp(self):
        self.text = COMPOSE.read_text()

    def test_every_clip_the_camera_publishes_is_committed(self):
        """`-i /clips/<name>` resolves through the ../../data/test_clips mount."""
        clips = re.findall(r"-i\s+/clips/([\w.\-]+)", self.text)
        self.assertTrue(clips, "no clip found in the compose file — did the camera change?")
        for name in clips:
            rel = f"data/test_clips/{name}"
            self.assertTrue((ROOT / rel).exists(), f"{rel} does not exist")
            self.assertTrue(_tracked(rel),
                            f"{rel} is not tracked by git — it exists on this machine only, "
                            f"so CI will decode nothing and fail as an RTSP 404")

    def test_every_mounted_host_path_exists_in_the_repo(self):
        """A missing bind mount is created as an empty directory by Docker,
        which fails silently rather than loudly."""
        for host in re.findall(r"^\s+-\s+(\.[\w./\-]+):", self.text, re.M):
            p = (COMPOSE.parent / host).resolve()
            if p == ROOT or ROOT not in p.parents:
                continue                      # the whole-repo mount, and ./out (generated)
            if p.name == "out":
                continue                      # written by the run itself
            self.assertTrue(p.exists(), f"compose mounts {host}, which does not exist")

    def test_the_site_config_the_engine_loads_is_committed(self):
        for cfg in re.findall(r"--site-config\s+(\S+)", self.text):
            rel = "tests/e2e/site" + cfg.split("/site", 1)[1]
            self.assertTrue(_tracked(rel), f"{rel} is not tracked by git")


if __name__ == "__main__":
    unittest.main()
