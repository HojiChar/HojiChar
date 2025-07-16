import pytest

from hojichar.core.models import Document, Statistics, Token


def test_repr() -> None:
    doc = Document("test", extras={"test": "test"})
    assert repr(doc) == "Document(text='test', is_rejected=False, extras={'test': 'test'})"
    assert eval(repr(doc)).text == doc.text
    assert eval(repr(doc)).extras == doc.extras


# --- Token ---
def test_token_defaults_and_str():
    t = Token("hello")
    assert t.text == "hello"
    assert t.original == "hello"
    assert not t.is_rejected
    assert str(t) == "hello"


def test_token_rejected_flag():
    t = Token("foo", is_rejected=True)
    assert t.is_rejected


# --- Document ---
def test_document_defaults():
    doc = Document("sample")
    assert doc.text == "sample"
    assert doc.original == "sample"
    assert not doc.is_rejected
    assert doc.tokens == []
    assert doc.extras == {}
    assert doc.reject_reason == {}
    assert str(doc) == "sample"
    # __repr__ contains key fields
    rep = repr(doc)
    assert "Document(text='sample'" in rep and "is_rejected=False" in rep


def test_document_with_tokens_and_extras():
    tok_list = [Token("a"), Token("b", True)]
    extras = {"foo": 123}
    doc = Document("text", is_rejected=True, tokens=tok_list, extras=extras)
    # 引数をそのまま保持する
    assert doc.tokens is tok_list
    assert doc.extras is extras
    assert doc.is_rejected is True


def test_deprecated_set_get_tokens():
    doc = Document("x")
    # set_tokens / get_tokens が Token オブジェクトを正しく扱うか
    doc.set_tokens(["alpha", "beta"])
    assert all(isinstance(t, Token) for t in doc.tokens)
    assert doc.get_tokens() == ["alpha", "beta"]


# --- Statistics ---
@pytest.fixture
def stat1():
    return Statistics(
        name="layer1",
        input_num=1,
        input_bytes=100,
        input_chars=10,
        output_num=1,
        output_bytes=80,
        output_chars=8,
        discard_num=0,
        diff_bytes=-20,
        diff_chars=-2,
        cumulative_time_ns=50,
    )


@pytest.fixture
def stat2():
    return Statistics(
        name="layer1",
        input_num=2,
        input_bytes=200,
        input_chars=20,
        output_num=1,
        output_bytes=150,
        output_chars=15,
        discard_num=1,
        diff_bytes=-50,
        diff_chars=-5,
        cumulative_time_ns=100,
    )


def test_statistics_update_and_reset(stat1, stat2):
    s = stat1
    s.update(stat2)
    # 足し算が正しく行われる
    assert s.input_num == 3
    assert s.input_bytes == 300
    assert s.input_chars == 30
    assert s.output_num == 2
    assert s.output_bytes == 230
    assert s.output_chars == 23
    assert s.discard_num == 1
    assert s.diff_bytes == -70
    assert s.diff_chars == -7
    assert s.cumulative_time_ns == 150

    # リセット後は全てゼロに
    s.reset()
    for field in (
        "input_num",
        "input_bytes",
        "input_chars",
        "output_num",
        "output_bytes",
        "output_chars",
        "discard_num",
        "diff_bytes",
        "diff_chars",
        "cumulative_time_ns",
    ):
        assert getattr(s, field) == 0


def test_statistics_add_and_assertion(stat1, stat2):
    # 正常ケース
    combined = Statistics.add(stat1, stat2)
    assert combined.name == "layer1"
    assert combined.input_num == 3
    # 名前が異なる場合は AssertionError
    stat_other = Statistics(name="other")
    with pytest.raises(AssertionError):
        Statistics.add(stat1, stat_other)


def test_statistics_add_list_of_stats_and_mismatch(stat1, stat2):
    a = [stat1]
    b = [stat2]
    # 正常ケース
    result = Statistics.add_list_of_stats(a, b)
    assert isinstance(result, list) and len(result) == 1
    assert result[0].input_num == 3

    # 名前の集合が異なる場合は ValueError
    c = [Statistics(name="diff")]
    with pytest.raises(ValueError):
        Statistics.add_list_of_stats(a, c)


def test_statistics_get_filter_and_key_error(stat1):
    stats = [stat1]
    found = Statistics.get_filter("layer1", stats)
    assert found is stat1

    with pytest.raises(KeyError):
        Statistics.get_filter("not_exist", stats)
