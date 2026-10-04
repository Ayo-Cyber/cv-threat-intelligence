"""Email alerts: the spec parses, the message is a real HTML mail with the
evidence inline, SMTP is driven correctly, and a half-configured channel
never pretends to work."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.serving.alert_sink import ConsoleNotifier, MultiNotifier, build_notifier
from cvti.serving.notify_email import EmailNotifier, parse_email_spec, render_html

SPEC = "email:" + urlencode({
    "host": "smtp.example.com", "port": "587", "user": "ops@site.com",
    "password": "p:ss,w@rd", "from": "argus@site.com",
    "to": "a@x.com;b@y.com", "security": "starttls",
})
EVENT = {
    "ts": 1.0, "iso": "2026-10-04T09:30:00", "id": 7, "camera_id": "Forecourt ATM",
    "rule": "video_theft_candidate", "priority": "high", "confidence": 0.95,
    "zone": None, "track_id": 3, "object_label": None,
    "reason": "CONFIRMED: a person is forcing the ATM panel with a tool <b>",
    "evidence_dir": None, "link": "https://argus.local/alert/7",
}


class SpecTest(unittest.TestCase):
    def test_passwords_with_colons_and_recipients_with_commas_survive(self):
        n = build_notifier(SPEC)
        self.assertIsInstance(n, EmailNotifier)
        self.assertEqual(n.password, "p:ss,w@rd")
        self.assertEqual(n.recipients, ["a@x.com", "b@y.com"])
        self.assertEqual((n.host, n.port, n.security), ("smtp.example.com", 587, "starttls"))
        n.close()

    def test_incomplete_settings_do_not_pretend_to_work(self):
        for body in ("", "host=smtp.x.com", "host=smtp.x.com&from=a@x.com", "from=a@x.com&to=b@y.com"):
            with self.subTest(body=body):
                self.assertIsNone(parse_email_spec(body))
                self.assertIsInstance(build_notifier("email:" + body), ConsoleNotifier)

    def test_from_defaults_to_the_login_and_port_to_the_security_mode(self):
        s = parse_email_spec(urlencode({"host": "h", "user": "u@x.com", "to": "a@x.com", "security": "ssl"}))
        self.assertEqual((s["sender"], s["port"]), ("u@x.com", 465))

    def test_alongside_telegram_in_one_spec(self):
        n = build_notifier("console,telegram:1:AAH:99," + SPEC)
        self.assertIsInstance(n, MultiNotifier)
        self.assertTrue(any(isinstance(x, EmailNotifier) for x in n.notifiers))
        n.close()


class MessageTest(unittest.TestCase):
    def _notifier(self):
        return EmailNotifier("smtp.example.com", 587, "ops@site.com", "pw", "argus@site.com",
                             ["a@x.com"], background=False)

    def test_html_is_escaped_and_carries_the_facts(self):
        page = render_html(EVENT, ["img1"])
        self.assertIn("VIDEO THEFT", page)
        self.assertIn("Forecourt ATM", page)
        self.assertIn("2026-10-04 09:30:00", page)
        self.assertIn("95%", page)
        self.assertIn('src="cid:img1"', page)
        self.assertIn('href="https://argus.local/alert/7"', page)
        self.assertIn("&lt;b&gt;", page)                 # the reason is escaped
        self.assertNotIn("<b>", page.split("<body")[1])  # not injected as markup
        self.assertNotIn("<script", page.lower())

    def test_message_has_text_and_html_with_the_frames_inline(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = Path(tmp) / "ev"; ev.mkdir()
            (ev / "frame_000.jpg").write_bytes(b"\xff\xd8fake")
            (ev / "subject.jpg").write_bytes(b"\xff\xd8subject")
            msg = self._notifier().build_message({**EVENT, "evidence_dir": str(ev)})
        self.assertEqual(msg["Subject"], "[Argus] HIGH — VIDEO THEFT on Forecourt ATM")
        self.assertEqual(msg["To"], "a@x.com")
        types = [p.get_content_type() for p in msg.walk()]
        self.assertIn("text/plain", types)
        self.assertIn("text/html", types)
        self.assertEqual(types.count("image/jpeg"), 2)
        html_part = next(p for p in msg.walk() if p.get_content_type() == "text/html")
        body = html_part.get_content()
        images = [p for p in msg.walk() if p.get_content_type() == "image/jpeg"]
        for img in images:
            self.assertIn(f'cid:{img["Content-ID"].strip("<>")}', body)
        # the subject shot leads: it is the picture a phone shows first
        self.assertEqual(images[0].get_filename(), "subject.jpg")

    def test_no_evidence_yet_is_still_a_complete_message(self):
        msg = self._notifier().build_message(EVENT)
        self.assertNotIn("image/jpeg", [p.get_content_type() for p in msg.walk()])
        self.assertIn("No evidence frame yet", msg.get_body(("html",)).get_content())


class DeliveryTest(unittest.TestCase):
    def test_starttls_login_and_send(self):
        server = MagicMock()
        server.__enter__ = MagicMock(return_value=server)
        server.__exit__ = MagicMock(return_value=False)
        with patch("cvti.serving.notify_email.smtplib.SMTP", return_value=server) as smtp:
            n = EmailNotifier("smtp.example.com", 587, "ops@site.com", "pw", "argus@site.com",
                              ["a@x.com", "b@y.com"], background=False)
            n.notify(EVENT)
        smtp.assert_called_once_with("smtp.example.com", 587, timeout=15.0)
        server.starttls.assert_called_once()
        server.login.assert_called_once_with("ops@site.com", "pw")
        self.assertEqual(server.send_message.call_count, 1)
        sent = server.send_message.call_args[0][0]
        self.assertEqual(sent["To"], "a@x.com, b@y.com")
        self.assertEqual(n.sent, 1)
        self.assertEqual(n.last_error, "")

    def test_ssl_mode_uses_smtp_ssl_and_no_starttls(self):
        server = MagicMock()
        server.__enter__ = MagicMock(return_value=server)
        server.__exit__ = MagicMock(return_value=False)
        with patch("cvti.serving.notify_email.smtplib.SMTP_SSL", return_value=server) as smtp:
            EmailNotifier("smtp.example.com", 465, "", "", "argus@site.com", ["a@x.com"],
                          security="ssl", background=False).notify(EVENT)
        smtp.assert_called_once()
        server.starttls.assert_not_called()
        server.login.assert_not_called()          # no username: no login
        server.send_message.assert_called_once()

    def test_a_failure_is_logged_and_remembered_never_raised(self):
        import smtplib
        with patch("cvti.serving.notify_email.smtplib.SMTP",
                   side_effect=smtplib.SMTPAuthenticationError(535, b"bad credentials")):
            n = EmailNotifier("smtp.example.com", 587, "u", "wrong", "argus@site.com", ["a@x.com"],
                              background=False)
            n.notify(EVENT)                        # must not raise into the gate
        self.assertIn("SMTPAuthenticationError", n.last_error)
        self.assertEqual(n.sent, 0)

    def test_background_queue_delivers_and_close_drains(self):
        server = MagicMock()
        server.__enter__ = MagicMock(return_value=server)
        server.__exit__ = MagicMock(return_value=False)
        with patch("cvti.serving.notify_email.smtplib.SMTP", return_value=server):
            n = EmailNotifier("smtp.example.com", 587, "", "", "argus@site.com", ["a@x.com"])
            n.notify(EVENT); n.notify(EVENT)
            n.close()
        self.assertEqual(server.send_message.call_count, 2)


class TestAlertReportsSmtpFailuresTest(unittest.TestCase):
    def test_send_test_notification_surfaces_the_channel_error(self):
        import smtplib
        from cvti.app.console_backend import ConsoleBackend
        from cvti.serving import onboarding
        be = ConsoleBackend.__new__(ConsoleBackend)
        be.site_path = "unused.json"
        with patch.object(onboarding, "get_site_meta", return_value={"notify": SPEC}), \
             patch("cvti.serving.notify_email.smtplib.SMTP",
                   side_effect=smtplib.SMTPAuthenticationError(535, b"bad credentials")):
            out = be.send_test_notification()
        self.assertFalse(out["ok"])
        self.assertIn("SMTPAuthenticationError", out["error"])


if __name__ == "__main__":
    unittest.main()
