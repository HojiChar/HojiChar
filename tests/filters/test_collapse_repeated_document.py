import pytest

from hojichar import Document
from hojichar.filters.document_filters import CollapseRepeatedDocument

# 49 characters.
BLOCK = "HojiChar はテキストの前処理パイプラインを組み立てるための Python ライブラリです。"
PAGE = (
    "お知らせ\n"
    "令和6年度の図書館の開館時間を変更します。\n"
    "平日は午前9時から午後8時まで、土日祝日は午前9時から午後5時までです。\n"
    "詳しくは図書館のウェブサイトをご覧ください。"
)


@pytest.mark.parametrize("separator", ["", " ", "\n", "\n\n", "\r\n", " | ", "---", "\n\n\n\n"])
def test_two_copies(separator: str) -> None:
    doc = CollapseRepeatedDocument().apply(Document(BLOCK + separator + BLOCK))
    assert doc.text == BLOCK
    assert not doc.is_rejected


@pytest.mark.parametrize(
    "text",
    [
        BLOCK * 3,
        BLOCK + "\n" + BLOCK + "\n" + BLOCK,
        BLOCK + "\n\n" + BLOCK + "\n\n" + BLOCK,
        BLOCK + "\n" + BLOCK + "\n\n" + BLOCK,  # separators of different lengths
        BLOCK + " / " + BLOCK + " / " + BLOCK,
    ],
)
def test_three_copies(text: str) -> None:
    assert CollapseRepeatedDocument().apply(Document(text)).text == BLOCK


def test_multiline_page() -> None:
    doc = CollapseRepeatedDocument().apply(Document(PAGE + "\n" + PAGE + "\n"))
    assert doc.text == PAGE


def test_surrounding_whitespace_is_removed() -> None:
    doc = CollapseRepeatedDocument().apply(Document("\n  " + BLOCK + "\n" + BLOCK + "  \n"))
    assert doc.text == BLOCK


@pytest.mark.parametrize(
    "text",
    [
        "",
        BLOCK,
        PAGE,
        # Too short to be a doubled page.
        "短い文です。" * 2,
        BLOCK[:39] + "\n" + BLOCK[:39],
        # The separator is too long.
        BLOCK + "-----" + BLOCK,
        BLOCK + "【続きを読む】" + BLOCK,
        # The copies differ.
        BLOCK + "\n" + BLOCK.replace("Python", "python"),
        # Other text follows the copies.
        BLOCK + "\n" + BLOCK + "\n以上です。",
        "はじめに。" + BLOCK + "\n" + BLOCK,
        BLOCK + "\n" + BLOCK + "\n" + BLOCK + "\n以上です。",
    ],
)
def test_unchanged(text: str) -> None:
    doc = CollapseRepeatedDocument().apply(Document(text))
    assert doc.text == text
    assert not doc.is_rejected


def test_min_block_len() -> None:
    text = "短い文です。\n短い文です。"
    assert CollapseRepeatedDocument().apply(Document(text)).text == text
    assert CollapseRepeatedDocument(min_block_len=5).apply(Document(text)).text == "短い文です。"


def test_kept_block_is_at_least_min_block_len() -> None:
    # 30 characters on each side of a long whitespace separator: 40 characters with the spaces,
    # but the block itself is shorter than `min_block_len`.
    block = BLOCK[:30]
    text = block + " " * 20 + block
    assert CollapseRepeatedDocument().apply(Document(text)).text == text


def test_max_separator_len() -> None:
    text = BLOCK + "【続きを読む】" + BLOCK
    assert CollapseRepeatedDocument(max_separator_len=7).apply(Document(text)).text == BLOCK
