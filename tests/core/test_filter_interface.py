import pytest

from hojichar.core.filter_interface import Filter
from hojichar.core.models import Document


class DummyFilter(Filter):
    """テスト用の単純フィルタ: 文字列末尾に "_ok" を付与"""

    def apply(self, document: Document) -> Document:
        document.text = document.text + "_ok"
        return document


def test_apply_filter_deprecation():
    filt = DummyFilter()
    doc = Document("hello")
    # apply_filter() で DeprecationWarning が出ること
    with pytest.warns(DeprecationWarning):
        out = filt.apply_filter(doc)
    assert isinstance(out, Document)
    assert out.text == "hello_ok"


def test_apply_batch_accepts_tuple_sequence():
    filt = DummyFilter(p=1.0)
    docs: tuple[Document, ...] = (Document("a"), Document("b"))
    out = filt.apply_batch(docs)
    assert isinstance(out, list)
    assert [d.text for d in out] == ["a_ok", "b_ok"]


def test__apply_batch_respects_p():
    docs = [Document("x"), Document("y")]
    # p=0 → まったく適用されない
    filt0 = DummyFilter(p=0.0, random_state=0)
    out0 = filt0._apply_batch(docs)
    assert [d.text for d in out0] == ["x", "y"]
    # p=1 → 常に適用される
    filt1 = DummyFilter(p=1.0)
    out1 = filt1._apply_batch(docs)
    assert [d.text for d in out1] == ["x_ok", "y_ok"]


def test_record_stats_and_diff_stats_logic():
    filt = DummyFilter()
    d1 = Document("hi")
    stats1 = filt.record_stats(d1)
    # 非破棄ケース: バイト数／文字数が +3、time が +100
    stats2 = {
        "is_rejected": False,
        "bytes": stats1["bytes"] + 3,
        "num_chars": stats1["num_chars"] + 3,
        "time_ns": stats1["time_ns"] + 100,
    }
    diff = filt.diff_stats(stats1, stats2)
    assert diff["num_discard"] == 0
    assert diff["diff_bytes"] == 3
    assert diff["diff_chars"] == 3
    assert diff["cumulative_time_ns"] == 100

    # 破棄ケース: is_rejected が False→True に
    stats3 = {
        "is_rejected": True,
        "bytes": stats1["bytes"],  # 無視される
        "num_chars": stats1["num_chars"],
        "time_ns": stats1["time_ns"] + 50,
    }
    diff2 = filt.diff_stats(stats1, stats3)
    assert diff2["num_discard"] == 1
    assert diff2["diff_bytes"] == -stats1["bytes"]
    assert diff2["diff_chars"] == -stats1["num_chars"]
    assert diff2["cumulative_time_ns"] == 50


def test_get_statistics_accumulates_across_calls():
    filt = DummyFilter(p=1.0)
    # 2 件処理すると、_ok で+3 bytes×2、+3 chars×2
    for txt in ["A", "BB"]:
        filt._apply(Document(txt))
    stats = filt.get_statistics()
    assert stats["num_discard"] == 0
    assert stats["diff_bytes"] == 6
    assert stats["diff_chars"] == 6


def test_probability_reproducibility_with_seed():
    seed = 2025
    filt1 = DummyFilter(p=0.3, random_state=seed)
    filt2 = DummyFilter(p=0.3, random_state=seed)
    out1 = [filt1("t") for _ in range(50)]
    out2 = [filt2("t") for _ in range(50)]
    assert out1 == out2


def test_shared_rng_instance_and_state_via_stub():
    class DummyFilter(Filter):
        def apply(self, document: Document) -> Document:
            return document

    # カウントアップするだけの簡易スタブ RNG
    class StubRNG:
        def __init__(self):
            self.counter = 0

        def random(self):
            self.counter += 1
            return self.counter

    stub = StubRNG()
    f1 = DummyFilter(p=0.5, random_state=None)
    f2 = DummyFilter(p=0.5, random_state=None)

    # 両フィルタに同じスタブをセット
    f1.set_rng_if_not_initialized(stub)
    f2.set_rng_if_not_initialized(stub)

    # identity が同じこと
    assert f1._rng is stub
    assert f2._rng is stub

    # f1 で一回消費
    assert f1._rng.random() == 1
    # f2 で二回目を消費
    assert f2._rng.random() == 2
    # 直接 stub でも三回目を消費できる
    assert stub.random() == 3


def test_skip_rejected_in_batch_stream():
    # use_batch=True のときに skip_rejected=True が働くか
    docs = [Document("z") for _ in range(3)]
    for d in docs:
        d.is_rejected = True

    filt = DummyFilter(p=1.0, skip_rejected=True, use_batch=True, batch_size=2)
    out = list(filt.apply_stream(docs))
    # 全件スキップなので末尾 "_ok" は付かない
    assert all(d.text == "z" for d in out)

    filt2 = DummyFilter(p=1.0, skip_rejected=False, use_batch=True, batch_size=2)
    out2 = list(filt2.apply_stream(docs))
    assert all(d.text == "z_ok" for d in out2)
