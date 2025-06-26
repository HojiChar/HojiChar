import pytest

from hojichar.core.filter_interface import Filter
from hojichar.core.models import Document


class DummyFilter(Filter):
    """テスト用の単純フィルタ: 文字列末尾に "_ok" を付与"""

    def apply(self, document: Document) -> Document:
        document.text = document.text + "_ok"
        return document


class DummyAppendFilter(Filter):
    """document.text に固定文字を追加するだけのフィルタ"""

    def apply(self, document: Document) -> Document:
        document.text = document.text + "X"
        return document


class DummyRejectFilter(Filter):
    """常に is_rejected=True にして捨てるフィルタ"""

    def apply(self, document: Document) -> Document:
        document.is_rejected = True
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


def test_get_statistics_accumulates_across_calls():
    filt = DummyFilter(p=1.0)
    # 2 件処理すると、_ok で+3 bytes×2、+3 chars×2
    for txt in ["A", "BB"]:
        filt._apply(Document(txt))
    stats = filt.get_statistics()
    assert stats.discard_num == 0
    assert stats.diff_bytes == 6
    assert stats.diff_chars == 6


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
    f1._set_rng_if_not_initialized(stub)
    f2._set_rng_if_not_initialized(stub)

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


def test_single_apply_updates_statistics():
    f = DummyAppendFilter()
    # 初期 stats は全ゼロ
    assert f.get_statistics().to_dict()["input_num"] == 0

    out = f("AB")  # __call__ → _apply → apply
    assert out == "ABX"

    stats = f.get_statistics().to_dict()
    # 1 件処理された
    assert stats["input_num"] == 1
    assert stats["output_num"] == 1
    # 文字数・バイト数差分は +1
    assert stats["diff_chars"] == 1
    assert stats["diff_bytes"] == 1
    # 破棄なし
    assert stats["discard_num"] == 0


def test_batch_apply_updates_statistics():
    f = DummyAppendFilter()
    docs = [Document(""), Document("A")]
    # _apply_batch を使わないと統計が取れないので明示的に呼び出し
    processed = f._apply_batch(docs)
    # テキスト変換確認
    assert [d.text for d in processed] == ["X", "AX"]

    stats = f.get_statistics().to_dict()
    # 2 件入力、2 件出力
    assert stats["input_num"] == 2
    assert stats["output_num"] == 2
    # 合計文字差分 = +2
    assert stats["diff_chars"] == 2
    assert stats["diff_bytes"] == 2


def test_stream_apply_updates_statistics_and_cleans_extras():
    f = DummyRejectFilter(use_batch=False)
    docs = [Document("foo"), Document("bar")]
    out = list(f.apply_stream(docs))
    # 全部捨てられるのでテキストは元のまま、is_rejected=True
    assert all(d.is_rejected for d in out)

    stats = f.get_statistics().to_dict()
    # 2 件入力、0 件出力、2 件破棄
    assert stats["input_num"] == 2
    assert stats["output_num"] == 0
    assert stats["discard_num"] == 2
    # diff_bytes は -(元バイト数の合計)
    total_bytes = len("foo".encode()) + len("bar".encode())
    assert stats["diff_bytes"] == -total_bytes
    # extras に一時キーが残っていない
    for d in out:
        assert "__start_ns" not in d.extras
        assert "__input_bytes" not in d.extras
        assert "__input_chars" not in d.extras


def test_stream_apply_in_batches_updates_statistics():
    # use_batch=True でバッチ処理ルートを通す
    f = DummyAppendFilter(use_batch=True, batch_size=2)
    docs = [Document("A"), Document("BC"), Document("DEF")]
    out = list(f.apply_stream(iter(docs)))
    # 全件に "X" が追加されている
    assert [d.text for d in out] == ["AX", "BCX", "DEFX"]

    stats = f.get_statistics().to_dict()
    # 3 件入力／出力、diff_bytes=3, diff_chars=3
    assert stats["input_num"] == 3
    assert stats["output_num"] == 3
    assert stats["discard_num"] == 0
    assert stats["diff_chars"] == 3
    assert stats["diff_bytes"] == 3
