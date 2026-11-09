import pytest

from hojichar import Document
from hojichar.filters.document_filters import SingleCharacterRepetitionFilter

# A long log with one separator line: the run is 200 characters in a 24,200-character text.
LONG_LOG_WITH_SEPARATOR = "テストの実行ログです。\n" * 2000 + "=" * 200


@pytest.mark.parametrize(
    "input_str,is_rejected",
    [
        ("あ" * 200, True),
        ("い" * 200, True),
        ("今日の天気は" + "い" * 200 + "いいねぇ〜ッ！", True),
        ("う" * 30, False),
        ("あいあいあいあいあいああいあいあいあいあいあ" * 30, False),
        ("", False),
        (LONG_LOG_WITH_SEPARATOR, True),
    ],
)
def test_single_character_repetition_filter(input_str: str, is_rejected: bool) -> None:
    filter = SingleCharacterRepetitionFilter()
    assert filter.apply(Document(input_str)).is_rejected == is_rejected


@pytest.mark.parametrize(
    "input_str,is_rejected",
    [
        # The run is a large part of the document.
        ("あ" * 200, True),
        ("今日の天気は" + "い" * 200 + "いいねぇ〜ッ！", True),
        ("テストの実行ログです。\n" * 100 + "=" * 200, True),  # 200 of 1,400 characters
        # The run is a small part of the document.
        (LONG_LOG_WITH_SEPARATOR, False),
        ("テストの実行ログです。\n" * 200 + "=" * 200, False),  # 200 of 2,600 characters
        # `threshold` still applies: short runs pass whatever their share.
        ("う" * 30, False),
        ("", False),
    ],
)
def test_single_character_repetition_filter_min_ratio(input_str: str, is_rejected: bool) -> None:
    filter = SingleCharacterRepetitionFilter(min_ratio=0.1)
    assert filter.apply(Document(input_str)).is_rejected == is_rejected


def test_single_character_repetition_filter_min_ratio_boundary() -> None:
    # 200 '=' in 1,000 characters is a ratio of exactly 0.2.
    doc = "ab" * 400 + "=" * 200
    assert SingleCharacterRepetitionFilter(min_ratio=0.2).apply(Document(doc)).is_rejected
    assert not SingleCharacterRepetitionFilter(min_ratio=0.21).apply(Document(doc)).is_rejected


def test_single_character_repetition_filter_positional_args_unchanged() -> None:
    # The second positional argument is still `p` of `Filter`.
    filter = SingleCharacterRepetitionFilter(200, 0.5)
    assert filter.p == 0.5
    assert filter.min_ratio == 0.0
