import pytest

from hojichar import Document
from hojichar.filters.document_filters import DiscardTooManyEndingEllipsis


@pytest.mark.parametrize(
    "input_str,is_rejected",
    [
        ("これはまともな文書です\n", False),
        ("今日は...\nあなたの運勢...\n占いましょう…\n", True),
        ("", False),
    ],
)
def test_discard_too_many_ending_ellipsis(input_str: str, is_rejected: bool) -> None:
    filter = DiscardTooManyEndingEllipsis()
    assert filter.apply(Document(input_str)).is_rejected == is_rejected
