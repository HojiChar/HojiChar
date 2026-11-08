import pytest

from hojichar import Document
from hojichar.filters.document_filters import DiscardBBSComments

# Three matches in 18 characters: a date, a weekday and "コメント".
BBS_LINE = "2024年1月5日(金) コメント\n"
PROSE_LINE = "今日は運動会でした。子どもたちは最後まで力いっぱい走りました。\n"


@pytest.mark.parametrize(
    "text",
    [
        # Markdown tables and hyphenated words used to match once per "-".
        "| 名前 | 値 |\n| --- | --- |\n" + "| a | 1 |\n" * 3 + "| b | 2 |\n" * 3,
        "state-of-the-art, end-to-end, well-known and up-to-date methods. " * 3,
        "-" * 100,
    ],
)
def test_hyphens_do_not_match(text: str) -> None:
    filt = DiscardBBSComments()
    assert filt.keyword_pat.findall(text) == []
    assert not filt.apply(Document(text)).is_rejected


def test_dates_with_hyphens_still_match() -> None:
    filt = DiscardBBSComments()
    assert filt.keyword_pat.findall("2024-01-05 と 2024/1/5") == ["2024-01-05", "2024/1/5"]
    assert filt.apply(Document("2024-01-05 投稿\n" * 8)).is_rejected


def test_default_decides_by_count() -> None:
    filt = DiscardBBSComments()
    assert not filt.apply(Document(BBS_LINE * 4)).is_rejected  # 12 matches
    assert filt.apply(Document(BBS_LINE * 5)).is_rejected  # 15 matches
    # A long document is rejected on the count alone.
    long_doc = (PROSE_LINE * 20 + BBS_LINE) * 5
    assert filt.apply(Document(long_doc)).is_rejected


def test_min_matches_per_1k_chars() -> None:
    filt = DiscardBBSComments(min_matches_per_1k_chars=15.0)
    # 15 matches in about 3,300 characters (4.6 per 1,000): kept.
    long_doc = (PROSE_LINE * 20 + BBS_LINE) * 5
    assert not filt.apply(Document(long_doc)).is_rejected
    # 15 matches in 90 characters: rejected.
    assert filt.apply(Document(BBS_LINE * 5)).is_rejected
    # The count condition still applies to short dense documents.
    assert not filt.apply(Document(BBS_LINE * 4)).is_rejected


def test_min_matches_per_1k_chars_boundary() -> None:
    # 15 matches in exactly 1,000 characters.
    text = "コメント" * 15
    text += "あ" * (1000 - len(text))
    assert DiscardBBSComments(min_matches_per_1k_chars=15.0).apply(Document(text)).is_rejected
    assert not DiscardBBSComments(min_matches_per_1k_chars=15.1).apply(Document(text)).is_rejected


def test_positional_args_unchanged() -> None:
    filt = DiscardBBSComments(14, 0.5)
    assert filt.max_allowed_num == 14
    assert filt.p == 0.5
    assert filt.min_matches_per_1k_chars == 0.0
