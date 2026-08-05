from __future__ import annotations

import json
import time
from multiprocessing.pool import Pool
from unittest.mock import Mock

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


class DelayFilter(hojichar.Filter):
    def apply(self, document: hojichar.Document) -> hojichar.Document:
        delay, text = document.text.split(":", maxsplit=1)
        time.sleep(float(delay))
        document.text = text
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


def test_parallel_preserves_input_order_when_ordered() -> None:
    documents = [
        hojichar.Document("0.2:slow"),
        hojichar.Document("0:fast-1"),
        hojichar.Document("0:fast-2"),
    ]
    filter = hojichar.Compose([DelayFilter()])

    with Parallel(filter, num_jobs=3, ordered=True) as pfilter:
        processed_docs = list(pfilter.imap_apply(iter(documents)))

    assert [doc.text for doc in processed_docs] == ["slow", "fast-1", "fast-2"]


@pytest.mark.parametrize(
    ("ordered", "expected_method", "unexpected_method"),
    [
        (False, "imap_unordered", "imap"),
        (True, "imap", "imap_unordered"),
    ],
)
def test_parallel_selects_pool_iterator(
    ordered: bool, expected_method: str, unexpected_method: str
) -> None:
    pool = Mock(spec=Pool)
    pool.imap.return_value = iter([])
    pool.imap_unordered.return_value = iter([])

    pfilter = Parallel(hojichar.Compose([]), ordered=ordered)
    pfilter._pool = pool
    pfilter._pid_stats = {}

    assert list(pfilter.imap_apply(iter([]))) == []
    getattr(pool, expected_method).assert_called_once()
    getattr(pool, unexpected_method).assert_not_called()


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


class SleepFilter(hojichar.Filter):
    def __init__(self, seconds: float, **kwargs):
        super().__init__(**kwargs)
        self.seconds = seconds

    def apply(self, document: hojichar.Document) -> hojichar.Document:
        time.sleep(self.seconds)
        return document


def test_max_in_flight_bounds_input_consumption() -> None:
    """
    max_in_flight 個を超えるドキュメントが「入力から取り出されたが未 yield」に
    ならないことを検証する。semaphore の不変条件なのでタイミングに依存しない。
    """
    window = 8
    produced = [0]

    def producer() -> hojichar.Document:
        for i in range(100):
            produced[0] += 1
            yield hojichar.Document(f"doc_{i}")

    filter = hojichar.Compose([SleepFilter(0.001)])
    consumed = 0
    with Parallel(filter, num_jobs=2, ordered=True, max_in_flight=window) as pfilter:
        for _ in pfilter.imap_apply(producer()):
            consumed += 1
            assert produced[0] - consumed <= window

    assert consumed == 100


@pytest.mark.parametrize("ordered", [False, True])
def test_max_in_flight_results_match_unbounded(ordered: bool) -> None:
    documents = [hojichar.Document(json.dumps({"text": f"doc_{i}"})) for i in range(20)]
    filter = hojichar.Compose([JSONLoader(), JSONDumper()])

    with Parallel(filter, num_jobs=2, ordered=ordered, max_in_flight=3) as pfilter:
        processed = list(pfilter.imap_apply(iter(documents)))

    assert set(str(doc) for doc in processed) == set(str(doc) for doc in documents)


def test_max_in_flight_early_exit_does_not_hang() -> None:
    """
    消費側が途中で iteration をやめても、gate で待機している feeder が
    解放され pool の shutdown が完了することを検証する。
    """
    documents = (hojichar.Document(f"doc_{i}") for i in range(1000))
    filter = hojichar.Compose([SleepFilter(0.001)])

    with Parallel(filter, num_jobs=2, max_in_flight=2) as pfilter:
        for doc in pfilter.imap_apply(documents):
            break
    # reaching here without a deadlock is the assertion


def test_max_in_flight_rejects_non_positive_values() -> None:
    filter = hojichar.Compose([JSONLoader()])
    with pytest.raises(ValueError):
        Parallel(filter, max_in_flight=0)


def test_max_in_flight_bound_is_strict() -> None:
    """
    厳密なバックプレッシャの回帰テスト。permit は「取得してから next() で
    取り出し、結果を呼び出し側へ渡し終える (次の next() が来る) まで保持」
    でなければならない。取得前に先読みする実装や、yield より前に release
    する実装では、window=1 でも 2 件目が source から取り出される。
    """
    produced = [0]

    def producer() -> hojichar.Document:
        for i in range(50):
            produced[0] += 1
            yield hojichar.Document(f"doc_{i}")

    filter = hojichar.Compose([DummyAppendFilter("")])
    with Parallel(filter, num_jobs=1, ordered=True, max_in_flight=1) as pfilter:
        iterator = pfilter.imap_apply(producer())
        next(iterator)  # doc_1 is handed to us and still holds its permit
        time.sleep(0.3)  # give the feeder ample time to advance as far as it can
        assert produced[0] == 1

        next(iterator)  # requesting doc_2 releases doc_1's permit
        time.sleep(0.3)
        assert produced[0] == 2


def test_max_in_flight_feeder_stops_after_abandoned_iteration() -> None:
    """
    window 枯渇で feeder が permit 待ちの間に消費側が iteration を打ち切った
    場合、クローズで返却された permit を feeder が拾って next() を余分に
    呼ばないことの回帰テスト。stop を立ててから permit を返す順序と、
    取得後の stop 再チェックの両方が必要になる。
    """
    produced = [0]

    def producer() -> hojichar.Document:
        for i in range(50):
            produced[0] += 1
            yield hojichar.Document(f"doc_{i}")

    filter = hojichar.Compose([DummyAppendFilter("")])
    with Parallel(filter, num_jobs=1, ordered=True, max_in_flight=1) as pfilter:
        iterator = pfilter.imap_apply(producer())
        next(iterator)  # doc_1 holds the only permit
        time.sleep(0.3)  # let the feeder park on the gate
        assert produced[0] == 1

        iterator.close()  # abandons iteration, returning doc_1's permit
        time.sleep(0.3)  # give a mis-woken feeder time to draw doc_2
        assert produced[0] == 1
