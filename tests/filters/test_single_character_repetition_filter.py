import pytest

from hojichar import Document
from hojichar.filters.document_filters import SingleCharacterRepetitionFilter


@pytest.mark.parametrize(
    "input_str,is_rejected",
    [
        ("あ" * 200, True),
        ("い" * 200, True),
        ("今日の天気は" + "い" * 200 + "いいねぇ〜ッ！", True),
        ("う" * 30, False),
        ("あいあいあいあいあいああいあいあいあいあいあ" * 30, False),
        ("", False),
    ],
)
def test_single_character_repetition_filter(input_str: str, is_rejected: bool) -> None:
    filter = SingleCharacterRepetitionFilter()
    assert filter.apply(Document(input_str)).is_rejected == is_rejected
