import pathlib

import hojichar
from hojichar.core.models import Document
from hojichar.filters.document_filters import NgWordsFilterEn, NgWordsFilterJa

BASE_PATH = pathlib.Path(hojichar.__path__[0])


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
