import pytest

from hojichar import Document
from hojichar.filters.document_filters import WordRepetitionRatioFilter


@pytest.mark.parametrize(
    "input_str,is_rejected",
    [
        (
            "派遣求人・派遣求人・派遣求人・派遣求人・派遣求人・派遣求人・派遣求人・派遣求人・派遣求人・派遣求人",
            True,
        ),
        (
            "合宿 免許 愛知などと検索した福津にお住いの方に合宿免許情報を紹介しています! | 全国合宿免許サイト情報\n合宿 免許 愛知などと検索した福津にお住いの方に合宿免許情報を紹介しています!",  # noqa: E501
            True,
        ),
        (
            "ウェブ\n本文: 【WOMB:8/15 水曜日】東京が持つローカルシーンと海外のリアルなシーンをX(融合)すべく立ち上がったパーティー【XPLODE】開催★平日の夜から盛り上がろう★渋谷ラウンジ★クーポン利用でお得! - クラブイベントサーチ\n【WOMB:8/15 水曜日】東京が持つローカルシーンと海外のリアルなシーンをX(融合)すべく立ち上がったパーティー【XPLODE】開催★平日の夜から盛り上がろう★渋谷ラウンジ★クーポン利用でお得!\n2018年08月15日(水)\t 22:00~",  # noqa: E501
            True,
        ),
        (
            "",
            False,
        ),
        ("2011年1月28日、EUがビスフェノールAを哺乳瓶に使用することを禁止", False),
        ("企業ログxを使って、あなたにピッタリな会社の企業情報を見つけてください！", False),
        ("本日のランチ情報をお知らせ致します", False),
    ],
)
def test_word_repetition_filter(input_str: str, is_rejected: bool) -> None:
    filter = WordRepetitionRatioFilter()
    assert filter.apply(Document(input_str)).is_rejected == is_rejected
