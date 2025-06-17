import pytest

from hojichar import Document
from hojichar.filters.document_filters import DiscardTooManySpecialToken


@pytest.mark.parametrize(
    "input_str,is_rejected",
    [
        (r"\\\\\\\\\\\\\\\\\\\\\\\////////////////////////////", True),
        (
            "この 章 に 規 定 する 手 続 を 遵 守 しないこと 以 外 の い か な る 理 由 によっても 国 際 登 録",
            True,
        ),
        ("おはよォ🤩👍👍👍👍", True),
        ("2011年1月28日、EUがビスフェノールAを哺乳瓶に使用することを禁止", False),
        ("企業ログxを使って、あなたにピッタリな会社の企業情報を見つけてください！", False),
        ("本日のランチ情報をお知らせ致します", False),
    ],
)
def test_discard_too_many_special_tokens(input_str: str, is_rejected: bool) -> None:
    filter = DiscardTooManySpecialToken()
    assert filter.apply(Document(input_str)).is_rejected == is_rejected
