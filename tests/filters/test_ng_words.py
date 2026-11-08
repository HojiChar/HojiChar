import pathlib

import pytest

import hojichar
from hojichar.core.models import Document
from hojichar.filters.document_filters import (
    DiscardAdultContentJa,
    DiscardDiscriminationContentJa,
    DiscardViolenceContentJa,
    NgWordsFilterEn,
    NgWordsFilterJa,
)

BASE_PATH = pathlib.Path(hojichar.__path__[0])

# 49 characters without NG words.
FILLER = "今日は朝から雨が降っていたので、図書館で本を読んで過ごしました。帰りに駅前の店で夕食をとりました。"


def test_ng_words_filter_ja():
    dict_path = BASE_PATH / "dict/dummy_ng_words.txt"
    ng_words_filter_ja = NgWordsFilterJa(dict_path, ignore_confused=False)
    assert ng_words_filter_ja.apply(Document("ほうじ茶")).is_rejected
    assert ng_words_filter_ja.apply(Document("ほ うじ茶")).is_rejected
    assert not ng_words_filter_ja.apply(Document("うじ茶")).is_rejected
    assert not ng_words_filter_ja.apply(Document("")).is_rejected
    assert ng_words_filter_ja.apply(Document("ラーメン")).is_rejected
    assert ng_words_filter_ja.apply(Document("ラーメンX")).is_rejected
    assert ng_words_filter_ja.apply(Document("Xラーメン")).is_rejected

    assert ng_words_filter_ja.apply(Document("ラララーメン")).is_rejected
    assert ng_words_filter_ja.apply(Document("ラーメンスープ")).is_rejected
    assert ng_words_filter_ja.apply(Document("ララーメンスープ")).is_rejected


def test_ng_words_filter_ja_ignore_confused():
    dict_path = BASE_PATH / "dict/dummy_ng_words.txt"
    ng_words_filter_ja = NgWordsFilterJa(dict_path, ignore_confused=True)
    assert ng_words_filter_ja.apply(Document("ほうじ茶")).is_rejected
    assert ng_words_filter_ja.apply(Document("ほ うじ茶")).is_rejected
    assert not ng_words_filter_ja.apply(Document("うじ茶")).is_rejected
    assert not ng_words_filter_ja.apply(Document("")).is_rejected
    assert ng_words_filter_ja.apply(Document("ラーメン")).is_rejected
    assert ng_words_filter_ja.apply(Document("ラーメンX")).is_rejected
    assert ng_words_filter_ja.apply(Document("Xラーメン")).is_rejected

    assert not ng_words_filter_ja.apply(Document("ラララーメン")).is_rejected
    assert not ng_words_filter_ja.apply(Document("ラーメンスープ")).is_rejected
    assert not ng_words_filter_ja.apply(Document("ララーメンスープ")).is_rejected


def test_ng_words_filter_en():
    dict_path = BASE_PATH / "dict/dummy_ng_words.txt"
    ng_words_filter_en = NgWordsFilterEn(dict_path)
    assert ng_words_filter_en.apply(Document("hojichar")).is_rejected
    assert ng_words_filter_en.apply(Document("h ojicha")).is_rejected
    assert not ng_words_filter_en.apply(Document("ojicha")).is_rejected
    assert not ng_words_filter_en.apply(Document("")).is_rejected

    # The English filter is not case-sensitive.
    assert ng_words_filter_en.apply(Document("Ramen")).is_rejected
    assert ng_words_filter_en.apply(Document("ramen")).is_rejected
    assert ng_words_filter_en.apply(Document("RAMEN")).is_rejected

    # In the English filter, words separated by a space, comma, or period will be filtered.
    assert not ng_words_filter_en.apply(Document("ramenx")).is_rejected
    assert not ng_words_filter_en.apply(Document("xramen")).is_rejected
    assert ng_words_filter_en.apply(Document("He eats ramen")).is_rejected
    assert ng_words_filter_en.apply(Document("He eats ramen.")).is_rejected
    assert ng_words_filter_en.apply(Document("He eats ramen, gyoza and fried rice.")).is_rejected
    assert ng_words_filter_en.apply(Document("Ramen is delicious.")).is_rejected
    assert not ng_words_filter_en.apply(Document("They are cameramen.")).is_rejected


def test_ng_words_filter_ja_count_and_density():
    dict_path = BASE_PATH / "dict/dummy_ng_words.txt"
    filt = NgWordsFilterJa(dict_path, max_allowed_num=2, min_matches_per_1k_chars=3.0)
    # Not more than 2 matches: kept, however dense.
    assert not filt.apply(Document("ラーメン")).is_rejected
    assert not filt.apply(Document("ラーメンとほうじ茶")).is_rejected
    # 3 matches in a short text: rejected.
    assert filt.apply(Document("ラーメンとほうじ茶とラーメン")).is_rejected
    # 3 matches in 2,070 characters (1.4 per 1,000): kept.
    long_doc = (FILLER * 14 + "ラーメン") * 3
    assert len(long_doc) == 2_070
    assert not filt.apply(Document(long_doc)).is_rejected
    # 3 matches in 600 characters (5 per 1,000): rejected.
    dense_doc = (FILLER * 4 + "ラーメン") * 3
    assert len(dense_doc) == 600
    assert filt.apply(Document(dense_doc)).is_rejected


def test_ng_words_filter_ja_count_only():
    dict_path = BASE_PATH / "dict/dummy_ng_words.txt"
    filt = NgWordsFilterJa(dict_path, max_allowed_num=2)
    assert not filt.apply(Document("ラーメンとほうじ茶")).is_rejected
    assert filt.apply(Document((FILLER * 100 + "ラーメン") * 3)).is_rejected


def test_ng_words_filter_ja_records_first_match():
    dict_path = BASE_PATH / "dict/dummy_ng_words.txt"
    filt = NgWordsFilterJa(dict_path, max_allowed_num=1)
    doc = filt.apply(Document("今日のおやつはほうじ茶とラーメンです"))
    assert doc.is_rejected
    assert filt.matched_text == "ほうじ茶"


def test_ng_words_filter_ja_ignore_confused_with_count():
    dict_path = BASE_PATH / "dict/dummy_ng_words.txt"
    filt = NgWordsFilterJa(dict_path, ignore_confused=True, max_allowed_num=1)
    # "ラーメンスープ" does not count with `ignore_confused`.
    assert not filt.apply(Document("ラーメンスープとラーメン")).is_rejected
    assert filt.apply(Document("ラーメンとラーメン")).is_rejected


@pytest.mark.parametrize(
    "filter_class,keyword",
    [
        (DiscardAdultContentJa, "<TEST_STRING_OF_ADULT_KEYWORD>"),
        (DiscardDiscriminationContentJa, "<TEST_STRING_OF_DISCRIMINATION_KEYWORD>"),
        (DiscardViolenceContentJa, "<TEST_STRING_OF_VIOLENCE_KEYWORD>"),
    ],
)
def test_ng_words_subclasses_accept_thresholds(filter_class, keyword):
    assert filter_class().apply(Document(keyword)).is_rejected

    filt = filter_class(ignore_confused=True, max_allowed_num=2, min_matches_per_1k_chars=3.0)
    assert not filt.apply(Document(FILLER + keyword)).is_rejected
    assert not filt.apply(Document((FILLER * 20 + keyword) * 3)).is_rejected
    assert filt.apply(Document((FILLER + keyword) * 3)).is_rejected


def test_ng_words_filter_ja_positional_args_unchanged():
    dict_path = BASE_PATH / "dict/dummy_ng_words.txt"
    filt = NgWordsFilterJa(dict_path, False, 0.5)
    assert filt.p == 0.5
    assert filt.max_allowed_num == 0
    assert filt.min_matches_per_1k_chars == 0.0
