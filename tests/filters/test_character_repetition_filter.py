import pytest

from hojichar import Document
from hojichar.filters.document_filters import CharRepetitionRatioFilter


@pytest.mark.parametrize(
    "input_str,is_rejected",
    [
        (
            "派遣求人・派遣求人・派遣求人・派遣求人・派遣求人・派遣求人・派遣求人・派遣求人・派遣求人・派遣求人",
            True,
        ),
        ("////////////////////////////////////////\\\\\\\\\\\\\\\\\\\\\\\\\\", True),
        (
            "愛知県の引越し見積もり・静岡県の引越し見積もり・長崎県の引越し見積もり・長野県の引越し見積もり・宮城県の引越し見積もり・福島県の引越し見積もり・秋田県の引越し見積もり・",
            True,
        ),
        ("2011年1月28日、EUがビスフェノールAを哺乳瓶に使用することを禁止", False),
        ("企業ログxを使って、あなたにピッタリな会社の企業情報を見つけてください！", False),
        ("本日のランチ情報をお知らせ致します", False),
    ],
)
def test_character_repetition_filter(input_str: str, is_rejected: bool) -> None:
    filter = CharRepetitionRatioFilter()
    assert filter.apply(Document(input_str)).is_rejected == is_rejected
