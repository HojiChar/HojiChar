from hojichar import Document
from hojichar.filters.document_filters import DiscardTooShortLines


def test_discard_too_short_lines() -> None:
    VALID_INPUT = "今日の天気は晴れのちくもり！！！！！いいですねぇ〜っ"

    INVALID_INPUT = """
    1
    2
    3
    4
    5
    6
    7
    """

    filter = DiscardTooShortLines()

    assert filter.apply(Document(VALID_INPUT)).is_rejected is False
    assert filter.apply(Document(INVALID_INPUT)).is_rejected
