"""Run the backend the way a customer's machine runs it — and make writes hurt.

Three field screenshots in two days, one disease: a path that resolves in this
repo and nowhere else. Feed configs (#45), feed data dirs (#49), zone rule
files (#50) — every fix was an instance; this harness is for the CLASS.

The simulation: the frozen flag is on, the user data dir is a sandbox, and the
process's WORKING DIRECTORY is read-only — exactly what a Start-Menu launch
into Program Files feels like. Every mutating flow the console offers is then
driven end to end. Any code that still builds a laptop-relative path dies
here with PermissionError instead of in a pilot's screenshot.
"""
from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cvti.app.console_backend import ConsoleBackend


class HostileInstallTest(unittest.TestCase):
    """Everything here runs with cwd read-only and sys.frozen = True."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.userdata = root / "userdata"
        self.userdata.mkdir()
        site = self.userdata / "site"
        site.mkdir()
        # the hostile part: a working directory nothing may write into
        self.jail = root / "program-files"
        self.jail.mkdir()
        self._cwd = os.getcwd()
        os.chmod(self.jail, stat.S_IRUSR | stat.S_IXUSR)
        os.chdir(self.jail)
        self._patches = [
            mock.patch.object(sys, "frozen", create=True, new=True),
            mock.patch("cvti.utils.user_data_dir", return_value=self.userdata),
        ]
        for p in self._patches:
            p.start()
        self.cb = ConsoleBackend(site_path=str(site / "site.json"),
                                 db_path=str(site / "events.db"), enable_demo=False)

    def tearDown(self):
        for p in self._patches:
            p.stop()
        os.chdir(self._cwd)
        os.chmod(self.jail, stat.S_IRWXU)
        self._tmp.cleanup()

    # -- the flows that produced field screenshots, plus every other mutation --

    def test_every_mutating_flow_survives_a_readonly_cwd(self):
        cb = self.cb
        r = cb.create_first_owner("owner", "pilot-pass-1")
        self.assertTrue(r.get("ok"), r)

        cams = cb.add_camera({"id": "cam1", "source": "rtsp://192.0.2.1/stream",
                              "config": "configs/all_threats_v1.json"})
        self.assertTrue(any(c["id"] == "cam1" for c in cams))

        z = cb.add_zone("cam1", "entrance", [[0, 0], [10, 0], [10, 10]], 16.0)
        self.assertNotIn("error", z, f"zone save failed in the jail: {z}")
        rules = json.loads(Path(z and self._rules_file()).read_text())["rules"]
        self.assertTrue(any(r0["name"] == "loitering_entrance" for r0 in rules))
        self.assertTrue(any(not r0["name"].startswith("loitering_") for r0 in rules),
                        "baseline preset vanished in the jail")

        cr = cb.add_custom_rule("cam1", "Is anyone climbing the gate?", 4.0)
        self.assertNotIn("error", cr, f"custom rule failed in the jail: {cr}")

        fd = cb._db_for_feed("demo", str(self.jail / "other.json"))
        self.assertTrue(fd.startswith(str(self.userdata)),
                        f"feed store escaped the user dir: {fd}")

        rm = cb.remove_zone("cam1", "entrance")
        self.assertNotIn("error", rm)

        # every artefact the session created lives in the user's dir
        written = [str(p) for p in self.userdata.rglob("*") if p.is_file()]
        self.assertTrue(written)
        self.assertEqual([p for p in self.jail.rglob("*")], [],
                         "something wrote into the read-only install dir")

    def _rules_file(self) -> str:
        cams = self.cb.list_cameras()
        return cams[0]["config"]

    def test_the_feed_registry_resolves_without_a_repo_cwd(self):
        reg = self.cb._feeds_registry()
        self.assertTrue(reg.get("sources"),
                        "no feeds offered — the registry read a cwd-relative path")

    def test_recovery_and_accounts_survive_the_jail(self):
        self.assertIn("auth_db", self.cb.auth_recovery())
        self.assertEqual(self.cb.auth_accounts()["users"], [])


if __name__ == "__main__":
    unittest.main()


class BackupCollectsWhatExistsTest(unittest.TestCase):
    """The pilot's Settings page showed 'Backup · 0 files · 0.3 KB' (29 Aug):
    the collector globbed cwd-relative configs/zones + configs/rules — empty
    or read-only on installs, and no longer where the app writes zones/rules
    anyway. A backup button that archives nothing is worse than none: it
    manufactures false safety."""

    def test_user_dir_zones_and_rules_are_captured(self):
        import json, zipfile
        from unittest import mock
        from cvti import backup as bk
        with tempfile.TemporaryDirectory() as tmp:
            ud = Path(tmp) / "userdata"
            (ud / "zones").mkdir(parents=True)
            (ud / "rules").mkdir(parents=True)
            (ud / "zones" / "cam1.json").write_text('{"zones": []}')
            (ud / "rules" / "cam1.json").write_text('{"rules": []}')
            site = Path(tmp) / "site.json"
            site.write_text('{"cameras": []}')
            with mock.patch("cvti.utils.user_data_dir", return_value=ud):
                out = bk.backup_config(site, dest_dir=Path(tmp) / "backups")
            self.assertTrue(out["ok"])
            self.assertGreaterEqual(out["entries"], 3,
                                    f"backup captured {out['entries']} entries — "
                                    "the pilot's zero-file zip again")
            names = zipfile.ZipFile(out["path"]).namelist()
            self.assertIn("configs/zones/cam1.json", names)
            self.assertIn("configs/rules/cam1.json", names)
