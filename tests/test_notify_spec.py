"""The notify spec must not silently do nothing.

Every channel here has a shape that looks configured but isn't:

  "telegram"   -- no token, no chat id. Falls through to console.
  "whatsapp"   -- credentials came from environment variables ONLY, which
                  nobody can set on a machine that runs an installer, so
                  ticking WhatsApp did nothing at all (pilot, 20 Sep).

Both failures are invisible: alerts keep flowing to the console and the
operator sees a configured channel that never delivers. The UI now refuses to
write a half-configured channel, and these pin the engine end of that deal.
"""
from __future__ import annotations

import unittest

from cvti.serving.alert_sink import (
    ConsoleNotifier,
    MultiNotifier,
    TelegramNotifier,
    WhatsAppNotifier,
    build_notifier,
)


class TelegramSpec(unittest.TestCase):
    def test_the_chat_id_is_the_part_after_the_LAST_colon(self):
        """A bot token contains a colon; splitting on the first one broke it."""
        notifier = build_notifier("telegram:123456789:AAHxyz:1883642843")
        self.assertIsInstance(notifier, TelegramNotifier)
        self.assertEqual(notifier.chat_id, "1883642843")
        self.assertTrue(notifier.base.endswith("/bot123456789:AAHxyz"))

    def test_a_group_chat_gets_the_slower_pace(self):
        private = build_notifier("telegram:1:AAH:1883642843")
        group = build_notifier("telegram:1:AAH:-1001234567890")
        self.assertLess(private.min_gap, group.min_gap,
                        "groups are rate-limited harder than private chats")

    def test_a_bare_telegram_is_not_a_telegram_channel(self):
        self.assertIsInstance(build_notifier("telegram"), ConsoleNotifier)


class WhatsAppSpec(unittest.TestCase):
    def test_it_can_carry_its_own_credentials(self):
        notifier = build_notifier("whatsapp:ACsid:token:+14155238886:+2348012345678")
        self.assertIsInstance(notifier, WhatsAppNotifier)
        self.assertEqual(notifier.to, "whatsapp:+2348012345678")
        self.assertEqual(notifier.from_, "whatsapp:+14155238886")
        self.assertIn("ACsid", notifier.url)

    def test_an_incomplete_spec_does_not_pretend_to_work(self):
        for spec in ("whatsapp:ACsid:token", "whatsapp:ACsid:token:+1:",
                     "whatsapp:::"):
            with self.subTest(spec=spec):
                self.assertIsInstance(build_notifier(spec), ConsoleNotifier)

    def test_bare_whatsapp_still_reads_the_environment(self):
        """The dev path keeps working; it just is not the only path any more."""
        import os
        keys = ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "WHATSAPP_TO")
        saved = {k: os.environ.get(k) for k in keys}
        try:
            for k, v in zip(keys, ("ACsid", "token", "+2348012345678")):
                os.environ[k] = v
            self.assertIsInstance(build_notifier("whatsapp"), WhatsAppNotifier)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


class SeveralChannels(unittest.TestCase):
    def test_console_and_telegram_and_whatsapp_together(self):
        notifier = build_notifier(
            "console,telegram:1:AAH:99,whatsapp:ACsid:token:+1:+234")
        self.assertIsInstance(notifier, MultiNotifier)
        kinds = {type(n).__name__ for n in notifier.notifiers}
        self.assertEqual(kinds, {"ConsoleNotifier", "TelegramNotifier",
                                 "WhatsAppNotifier"})

    def test_an_empty_spec_is_console(self):
        self.assertIsInstance(build_notifier(""), ConsoleNotifier)


if __name__ == "__main__":
    unittest.main()
