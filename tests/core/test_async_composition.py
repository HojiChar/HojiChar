from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from hojichar.core.async_composition import AsyncCompose, AsyncFilterAdapter
from hojichar.core.async_filter_interface import AsyncFilter
from hojichar.core.filter_interface import Filter
from hojichar.core.models import Document, Statistics


# Dummy synchronous Filter implementation
class SyncAppendFilter(Filter):
    def __init__(self, suffix: str, use_batch: bool = False, batch_size: int = 10):
        super().__init__(use_batch=use_batch, batch_size=batch_size)
        self.suffix = suffix

    def apply(self, document: Document) -> Document:
        document.text = document.text + self.suffix
        return document

    def apply_batch(self, batch):
        # Called when use_batch=True
        return [self.apply(doc) for doc in batch]


# Dummy asynchronous Filter implementation
class AsyncUpperFilter(AsyncFilter):
    async def apply(self, document: Document) -> Document:
        document.text = document.text.upper()
        return document


@pytest.mark.asyncio
async def test_adapter_apply():
    sync = SyncAppendFilter("_sync")
    adapter = AsyncFilterAdapter(sync_filter=sync)
    doc = Document(text="a")
    result = await adapter.apply(doc)
    assert result.text == "a_sync"


@pytest.mark.asyncio
async def test_adapter_apply_batch_use_batch_true():
    sync = SyncAppendFilter("_B", use_batch=True)
    adapter = AsyncFilterAdapter(sync_filter=sync)
    docs = [Document(text="d1"), Document(text="d2")]
    results = await adapter.apply_batch(docs)
    assert [d.text for d in results] == ["d1_B", "d2_B"]


@pytest.mark.asyncio
async def test_adapter_apply_batch_use_batch_false():
    sync = SyncAppendFilter("_N", use_batch=False)
    adapter = AsyncFilterAdapter(sync_filter=sync)
    docs = [Document(text="x"), Document(text="y")]
    results = await adapter.apply_batch(docs)
    assert [d.text for d in results] == ["x_N", "y_N"]


@pytest.mark.asyncio
async def test_adapter_shutdown_external_executor_not_shutdown():
    # Record thread IDs to ensure executor still usable after adapter.shutdown
    thread_ids = []

    def record_thread(doc: Document) -> Document:
        thread_ids.append(threading.get_ident())
        return doc

    class RecordFilter(SyncAppendFilter):
        def apply(self, document: Document) -> Document:
            return record_thread(document)

    sync = RecordFilter(suffix="", use_batch=False)
    executor = ThreadPoolExecutor(max_workers=1)
    adapter = AsyncFilterAdapter(sync_filter=sync, executor=executor)
    # Invoke apply to use executor
    doc = Document(text="test")
    await adapter.apply(doc)
    adapter.shutdown()
    # Executor should still accept new tasks
    fut = executor.submit(lambda: 123)
    assert fut.result() == 123
    executor.shutdown()


@pytest.mark.asyncio
async def test_async_compose_apply_and_batch_and_stream():
    # Set up filters: sync append, async uppercase, sync append
    sync1 = SyncAppendFilter("_1", use_batch=False)
    async_upper = AsyncUpperFilter()
    sync2 = SyncAppendFilter("_2", use_batch=False)

    comp = AsyncCompose(filters=[sync1, async_upper, sync2])

    # Test single apply
    doc = Document(text="t")
    out = await comp.apply(doc)
    # Order: t -> t_1 -> T_1 -> T_1_2
    assert out.text == "T_1_2"

    # Test batch apply
    docs = [Document(text="a"), Document(text="b")]
    outs = await comp.apply_batch(docs)
    assert [d.text for d in outs] == ["A_1_2", "B_1_2"]

    # Test stream apply
    stream_docs = [Document(text=s) for s in ["x", "y"]]
    results = []
    async for doc in comp.apply_stream(stream_docs):
        results.append(doc.text)
    assert results == ["X_1_2", "Y_1_2"]

    # Test statistics map length and names
    stats_map = comp.get_total_statistics_map()
    # Expect Total + three subfilters
    assert len(stats_map) == 4
    names = [m["name"] for m in stats_map]
    assert names[0] == "Total"


@pytest.mark.asyncio
async def test_compose_shutdown():
    # Ensure executors are shutdown only when owned
    sync = SyncAppendFilter("_")
    comp_external = AsyncCompose(filters=[sync], executor=ThreadPoolExecutor(max_workers=1))
    # External executor passed; after shutdown, executor remains usable
    exec_ref = comp_external._executor
    await comp_external.shutdown()
    fut = exec_ref.submit(lambda: 456)
    assert fut.result() == 456
    exec_ref.shutdown()

    # Internal executor owned; after shutdown, submitting should error
    comp_internal = AsyncCompose(filters=[sync])
    exec_int = comp_internal._executor
    await comp_internal.shutdown()
    with pytest.raises(RuntimeError):
        exec_int.submit(lambda: 0)


@pytest.mark.asyncio
async def test_statistics_after_apply():
    # Pipeline: append '_1', uppercase, append '_2'
    sync1 = SyncAppendFilter("_1")
    upper = AsyncUpperFilter()
    sync2 = SyncAppendFilter("_2")
    comp = AsyncCompose(filters=[sync1, upper, sync2])

    # Single document
    text = "abc"
    doc = Document(text=text)
    out = await comp.apply(doc)
    assert out.text == "ABC_1_2"

    # Gather statistics
    stats = comp.get_total_statistics()
    # Total stats is first
    total = stats[0]

    f1 = Statistics.get_filter("0-SyncAppendFilter", stats)
    f2 = Statistics.get_filter("1-AsyncUpperFilter", stats)
    f3 = Statistics.get_filter("2-SyncAppendFilter", stats)

    # Check first filter stats
    # input_bytes=3, output_bytes=5, diff_bytes=2
    assert f1.input_num == 1
    assert f1.input_bytes == len("abc")
    assert f1.output_bytes == len("abc" + "_1")
    assert f1.diff_bytes == len("_1")

    # Uppercase filter: no byte/char change
    assert f2.input_bytes == f2.output_bytes
    assert f2.diff_bytes == 0
    assert f2.input_num == 1

    # Third filter: append '_2'
    before_len = len((text + "_1"))
    after_len = len((text + "_1" + "_2"))
    assert f3.input_bytes == before_len
    assert f3.output_bytes == after_len
    assert f3.diff_bytes == len("_2")

    # Total stats: diff_bytes == sum of diffs
    expected_total_diff = f1.diff_bytes + f2.diff_bytes + f3.diff_bytes
    assert total.diff_bytes == expected_total_diff
    assert total.input_num == 1
    assert total.output_bytes == after_len


@pytest.mark.asyncio
async def test_statistics_after_batch_apply():
    sync1 = SyncAppendFilter("_1", use_batch=True)
    upper = AsyncUpperFilter()
    sync2 = SyncAppendFilter("_2", use_batch=True)
    comp = AsyncCompose(filters=[sync1, upper, sync2])

    # Two documents of equal length
    texts = ["aa", "bb"]
    docs = [Document(text=t) for t in texts]
    outs = await comp.apply_batch(docs)
    assert [d.text for d in outs] == [t.upper() + "_1_2" for t in texts]

    stats = comp.get_total_statistics()
    total = stats[0]

    f1 = Statistics.get_filter("0-SyncAppendFilter", stats)
    f2 = Statistics.get_filter("1-AsyncUpperFilter", stats)
    f3 = Statistics.get_filter("2-SyncAppendFilter", stats)

    # First filter: each doc diff len('_1')
    assert f1.input_num == 2
    assert f1.diff_bytes == 2 * len("_1")

    # Upper: no diff
    assert f2.diff_bytes == 0
    assert f2.input_num == 2

    # Third filter: each doc diff len('_2')
    assert f3.diff_bytes == 2 * len("_2")
    assert f3.input_num == 2

    # Total diff
    assert total.diff_bytes == f1.diff_bytes + f2.diff_bytes + f3.diff_bytes
    assert total.input_num == 2


@pytest.mark.asyncio
async def test_statistics_after_stream_apply():
    sync1 = SyncAppendFilter("_1")
    upper = AsyncUpperFilter()
    sync2 = SyncAppendFilter("_2")
    comp = AsyncCompose(filters=[sync1, upper, sync2])

    texts = ["x", "yy", "zzz"]
    docs = [Document(text=t) for t in texts]
    results = [doc.text async for doc in comp.apply_stream(docs)]
    assert results == [t.upper() + "_1_2" for t in texts]

    stats = comp.get_total_statistics()
    total = stats[0]
    f1 = Statistics.get_filter("0-SyncAppendFilter", stats)
    f2 = Statistics.get_filter("1-AsyncUpperFilter", stats)
    f3 = Statistics.get_filter("2-SyncAppendFilter", stats)

    # f1: sum of len('_1') per doc
    # assert f1.diff_bytes == len(texts) * len("_1")
    assert f1.input_num == len(texts)
    # f2 zero
    assert f2.diff_bytes == 0
    assert f2.input_num == len(texts)
    # f3
    assert f3.diff_bytes == len(texts) * len("_2")
    assert f3.input_num == len(texts)
    # total: sum
    assert total.diff_bytes == f1.diff_bytes + f2.diff_bytes + f3.diff_bytes
    assert total.input_num == len(texts)
