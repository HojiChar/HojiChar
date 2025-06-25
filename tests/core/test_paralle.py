from __future__ import annotations

import json

import pytest

import hojichar
from hojichar.core.parallel import Parallel
from hojichar.filters.document_filters import JSONDumper, JSONLoader


class RaiseKeywords(hojichar.Filter):
    def apply(self, document: hojichar.Document) -> hojichar.Document:
        text = document.text
        if "<raise>" in text:
            raise
        return document


class DummyAppendFilter(hojichar.Filter):
    """
    ドキュメントの text に self.suffix を追加するだけのフィルタ
    """

    def __init__(self, suffix: str, **kwargs):
        super().__init__(**kwargs)
        self.suffix = suffix

    def apply(self, document: hojichar.Document) -> hojichar.Document:
        document.text = document.text + self.suffix
        return document


@pytest.mark.parametrize("num_jobs", [1, 4, None])
def test_processed_docs_count(num_jobs: int | None) -> None:
    documents = [hojichar.Document(json.dumps({"text": f"doc_{i}"})) for i in range(10)]
    filter = hojichar.Compose([JSONLoader(), JSONDumper()])

    with Parallel(filter, num_jobs=num_jobs) as pfilter:
        list(pfilter.imap_apply(iter(documents)))
        assert pfilter.statistics_obj.total_info.processed_num == 10


@pytest.mark.parametrize("num_jobs", [1, 4, None])
def test_processed_docs_equality(num_jobs: int | None) -> None:
    documents = [hojichar.Document(json.dumps({"text": f"doc_{i}"})) for i in range(10)]
    filter = hojichar.Compose([JSONLoader(), JSONDumper()])

    with Parallel(filter, num_jobs=num_jobs) as pfilter:
        processed_docs = list(pfilter.imap_apply(iter(documents)))
        assert set(str(s) for s in processed_docs) == set(str(s) for s in documents)


@pytest.mark.parametrize("num_jobs", [1, 4, None])
def test_filter_statistics_increment(num_jobs: int | None) -> None:
    documents = [hojichar.Document(json.dumps({"text": f"doc_{i}"})) for i in range(10)]
    filter = hojichar.Compose([JSONLoader(), JSONDumper()])

    with Parallel(filter, num_jobs=num_jobs) as pfilter:
        list(pfilter.imap_apply(iter(documents)))

    assert filter.statistics_obj.total_info.processed_num == 10

    with Parallel(filter, num_jobs=num_jobs) as pfilter:
        list(pfilter.imap_apply(iter(documents)))

    assert filter.statistics_obj.total_info.processed_num == 20


@pytest.mark.parametrize("num_jobs", [1, 4, None])
def test_parallel_with_error_handling(num_jobs: int | None) -> None:
    documents = [hojichar.Document(f"<raise>_{i}") for i in range(10)]
    error_filter = hojichar.Compose([RaiseKeywords()])

    with pytest.raises(Exception):
        with Parallel(error_filter, num_jobs=num_jobs) as pfilter:
            list(pfilter.imap_apply(iter(documents)))
            pfilter.statistics_obj.total_info.processed_num == 0
    assert error_filter.statistics_obj.total_info.processed_num == 0

    with Parallel(error_filter, num_jobs=2, ignore_errors=True) as pfilter:
        processed_docs = list(pfilter.imap_apply(iter(documents)))
        assert list(str(s) for s in processed_docs) == [""] * 10
        pfilter.statistics_obj.total_info.processed_num == 0
    assert error_filter.statistics_obj.total_info.processed_num == 0


def test_parallel_statistics_collection():
    # 2 つのフィルタを持つ Compose を並列で適用
    f1 = DummyAppendFilter(suffix="1")
    f2 = DummyAppendFilter(suffix="2")
    comp = hojichar.Compose([f1, f2], random_state=0)

    docs = [hojichar.Document("A"), hojichar.Document("BB")]
    # num_jobs=2 にすると、2 つのワーカーがそれぞれ 1 件ずつ処理し、
    # pid ごとに統計が収集される
    with Parallel(comp, num_jobs=2, ignore_errors=False) as p:
        out = list(p.imap_apply(iter(docs)))

    # 出力文字列が正しく加工されている
    assert sorted([d.text for d in out]) == sorted(["A12", "BB12"])

    # 集約後の統計を取得
    stats = comp.get_total_statistics_map()
    total, layer0, layer1 = stats

    # --- Total 統計 ---
    assert total["name"] == "Total"
    # 2 ドキュメント入力・2 ドキュメント出力
    assert total["input_num"] == 2
    assert total["output_num"] == 2
    # 各フィルタ suffix 長さ 1 を 2 回（フィルタ×ドキュメント）適用 → 合計 +4
    assert total["diff_chars"] == 4
    assert total["diff_bytes"] == 4
    assert total["discard_num"] == 0

    # --- Layer0 (DummyAppendFilter 1) ---
    assert layer0["name"].startswith("0-")
    # このレイヤも 2 ドキュメントに適用
    assert layer0["input_num"] == 2
    assert layer0["output_num"] == 2
    # 1 文字を 2 ドキュメント分 → +2
    assert layer0["diff_chars"] == 2
    assert layer0["diff_bytes"] == 2

    # --- Layer1 (DummyAppendFilter 2) ---
    assert layer1["name"].startswith("1-")
    assert layer1["input_num"] == 2
    assert layer1["output_num"] == 2
    assert layer1["diff_chars"] == 2
    assert layer1["diff_bytes"] == 2
