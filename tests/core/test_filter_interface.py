import numpy as np
import pytest

from hojichar.core.filter_interface import Filter, _is_jsonable
from hojichar.core.models import Document


class DummyFilter(Filter):
    """テスト用の単純フィルタ: 文字列末尾に "_ok" を付与"""

    def apply(self, document: Document) -> Document:
        document.text = document.text + "_ok"
        return document


class ShutdownFilter(Filter):
    """shutdown() が呼ばれたかどうかを追跡するフィルタ"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.shutdown_called = False

    def apply(self, document: Document) -> Document:
        return document

    def shutdown(self):
        self.shutdown_called = True


def test_is_jsonable_primitives():
    assert _is_jsonable(None) is True
    assert _is_jsonable(True) is True
    assert _is_jsonable(123) is True
    assert _is_jsonable(3.14) is True
    assert _is_jsonable("hello") is True
    # list/dict はサポート対象外
    assert _is_jsonable([1, 2, 3]) is False
    assert _is_jsonable({"a": 1}) is False


def test_apply_and_call():
    filt = DummyFilter(p=1.0, skip_rejected=True, seed=0)
    # apply(document) で末尾に _ok が付く
    doc = Document("text")
    out = filt.apply(doc)
    assert isinstance(out, Document)
    assert out.text == "text_ok"
    # __call__ を使うと文字列が返る
    assert filt("abc") == "abc_ok"


def test_rng_init_conflict():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        DummyFilter(seed=0, rng=rng)


@pytest.mark.parametrize(
    "p,seed,text,expected_applications",
    [
        (1.0, 42, "t", 5),  # p=1.0 なら常に5回とも適用
        (0.0, 42, "t", 0),  # p=0.0 なら常に0回
    ],
)
def test_probability_skip(p, seed, text, expected_applications):
    filt = DummyFilter(p=p, seed=seed)
    # RNG 固定で5 回繰り返し
    results = [filt(text) for _ in range(5)]
    ok_count = sum(1 for r in results if r.endswith("_ok"))
    assert ok_count == expected_applications


def test_skip_rejected_flag_behavior():
    # is_rejected=True の場合は p=1.0 でも常にスキップ
    filt = DummyFilter(p=1.0, skip_rejected=True)
    doc = Document("orig")
    doc.is_rejected = True
    out = filt._apply(doc)
    assert out.text == "orig"  # 変更されず

    # skip_rejected=False にすると拒否済みも処理対象
    filt2 = DummyFilter(p=1.0, skip_rejected=False)
    doc2 = Document("orig")
    doc2.is_rejected = True
    out2 = filt2._apply(doc2)
    assert out2.text == "orig_ok"


def test_apply_batch_and_stream():
    # apply_batch: 新規ドキュメントリストで実行
    docs_batch = [Document(f"{i}") for i in range(5)]
    filt = DummyFilter(p=1.0, seed=0, use_batch=True, batch_size=2)
    batch_out = filt.apply_batch(docs_batch)
    assert [d.text for d in batch_out] == [f"{i}_ok" for i in range(5)]

    # apply_stream: 別リストで実行
    docs_stream = [Document(f"{i}") for i in range(5)]
    stream_out = list(filt.apply_stream(docs_stream))
    assert [d.text for d in stream_out] == [f"{i}_ok" for i in range(5)]

    # use_batch=False なら逐次適用
    docs_seq = [Document(f"{i}") for i in range(5)]
    filt2 = DummyFilter(p=1.0, use_batch=False)
    stream2 = list(filt2.apply_stream(docs_seq))
    assert [d.text for d in stream2] == [f"{i}_ok" for i in range(5)]


def test_get_jsonable_vars_excludes_private_and_nonjsonable():
    filt = DummyFilter(p=0.5, skip_rejected=False)
    # ダミーの追加属性を付与
    filt.foo = 123
    filt._bar = "hidden"
    filt.baz = [1, 2, 3]  # JSONable ではない
    vars_json = filt.get_jsonable_vars()
    assert "p" in vars_json
    assert vars_json["p"] == 0.5
    assert "foo" in vars_json
    assert "_bar" not in vars_json
    assert "baz" not in vars_json


def test_context_manager_shutdown():
    filt = ShutdownFilter()
    with filt as f:
        # enter で同一インスタンスが返る
        assert f is filt
    # with ブロック終了後に shutdown() が呼ばれている
    assert filt.shutdown_called is True
