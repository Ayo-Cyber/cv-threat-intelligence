"""The notify spec must survive a REAL Telegram token (it contains a colon).

'telegram:<token>:<chat>' with token='8691681982:AAH...' was split with
split(':', 2), which handed half the token to chat_id — no message could
ever leave. Found the first time a real token was wired (11 Sep).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.serving.alert_sink import TelegramNotifier, _build_one


def test_a_real_token_with_colon_parses_whole():
    n = _build_one("telegram:8691681982:AAHtokenSecondHalf:1883642843")
    assert isinstance(n, TelegramNotifier)
    assert n.base.endswith("/bot8691681982:AAHtokenSecondHalf")
    assert n.chat_id == "1883642843"


def test_a_negative_group_chat_id_parses(  ):
    n = _build_one("telegram:111:AAA:-1002233445566")
    assert n.chat_id == "-1002233445566"
