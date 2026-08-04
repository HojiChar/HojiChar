import asyncio

import pytest

from hojichar.core.async_filter_interface import AsyncFilter
from hojichar.core.models import Document


# Dummy filters for testing
class UppercaseFilter(AsyncFilter):
    async def apply(self, document: Document) -> Document:
        document.text = document.text.upper()
        return document


class RejectFirstFilter(AsyncFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._count = 0

    async def apply(self, document: Document) -> Document:
        # Reject first call, accept subsequent
        if self._count == 0:
            document.is_rejected = True
        self._count += 1
        return document


class CountingFilter(AsyncFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_count = 0

    async def apply(self, document: Document) -> Document:
        self.apply_count += 1
        return document


class ControlledFilter(AsyncFilter):
    def __init__(self, count: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.started = [asyncio.Event() for _ in range(count)]
        self.release = [asyncio.Event() for _ in range(count)]
        self.active = 0
        self.max_active = 0

    async def apply(self, document: Document) -> Document:
        index = int(document.text)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.started[index].set()
        try:
            await self.release[index].wait()
        finally:
            self.active -= 1
        return document


@pytest.mark.asyncio
async def test_call_and_apply():
    f = UppercaseFilter()
    text = "hello world"
    result_text = await f(text)
    assert result_text == text.upper()
    # Statistics map contains entry
    stats = f.get_statistics()
    assert stats.name == "UppercaseFilter"
    assert stats.input_chars == len(text)


@pytest.mark.asyncio
async def test_apply_batch():
    docs = [Document(text=s) for s in ["a", "b", "c"]]
    f = UppercaseFilter()
    out = await f.apply_batch(docs)
    assert [d.text for d in out] == ["A", "B", "C"]
    # Original docs updated in place or returned objects
    assert all(isinstance(d, Document) for d in out)


@pytest.mark.asyncio
async def test_apply_stream_sync_iterable():
    texts = ["one", "two", "three"]
    f = UppercaseFilter()
    # use_stream processes sync iterable
    results = []
    async for doc in f.apply_stream([Document(text=t) for t in texts]):
        results.append(doc.text)
    assert results == [s.upper() for s in texts]


@pytest.mark.asyncio
async def test_apply_stream_async_iterable():
    async def gen():
        for t in ["x", "y"]:
            yield Document(text=t)

    f = UppercaseFilter()
    results = [doc.text async for doc in f.apply_stream(gen())]
    assert results == ["X", "Y"]


@pytest.mark.asyncio
async def test_skip_rejected_behavior():
    docs = [Document(text="t1"), Document(text="t2")]
    # reject first, skip second
    f = RejectFirstFilter(skip_rejected=True)
    outputs = [doc async for doc in f.apply_stream(docs)]
    # First should be rejected, second not
    assert outputs[0].is_rejected
    assert not outputs[1].is_rejected
    # First reject_reason set, second reject_reason None
    assert isinstance(outputs[0].reject_reason, dict)


@pytest.mark.asyncio
async def test_probability_p():
    docs = [Document(text="a") for _ in range(5)]
    # p=0: apply should never run
    f0 = CountingFilter(p=0.0, random_state=42)
    _ = [doc async for doc in f0.apply_stream(docs)]
    assert f0.apply_count == 0
    # p=1: apply always runs
    f1 = CountingFilter(p=1.0, random_state=42)
    _ = [doc async for doc in f1.apply_stream(docs)]
    assert f1.apply_count == len(docs)


@pytest.mark.asyncio
async def test_use_batch_flag():
    docs = [Document(text=str(i)) for i in range(4)]
    # Batch of size 2
    f = UppercaseFilter(use_batch=True, batch_size=2)
    results = [doc.text async for doc in f.apply_stream(docs)]
    assert results == [s.upper() for s in ["0", "1", "2", "3"]]


@pytest.mark.asyncio
async def test_error_handling_in_apply():
    class ErrorFilter(AsyncFilter):
        async def apply(self, document: Document) -> Document:
            raise RuntimeError("fail")

    docs = [Document(text="t")]
    f = ErrorFilter()
    outputs = [doc async for doc in f.apply_stream(docs)]
    # Error should reject document
    assert outputs[0].is_rejected


@pytest.mark.asyncio
async def test_apply_stream_uses_sliding_window():
    docs = [Document(text=str(i)) for i in range(3)]
    f = ControlledFilter(count=3, batch_size=2)
    stream = f.apply_stream(docs)

    first_output = asyncio.create_task(stream.__anext__())
    await asyncio.wait_for(f.started[0].wait(), timeout=1)
    await asyncio.wait_for(f.started[1].wait(), timeout=1)

    f.release[0].set()
    assert (await asyncio.wait_for(first_output, timeout=1)).text == "0"

    # The next item starts without waiting for every item in the original window.
    second_output = asyncio.create_task(stream.__anext__())
    await asyncio.wait_for(f.started[2].wait(), timeout=1)
    assert not f.release[1].is_set()
    assert f.max_active == 2

    f.release[1].set()
    assert (await asyncio.wait_for(second_output, timeout=1)).text == "1"
    f.release[2].set()
    assert [doc.text async for doc in stream] == ["2"]


@pytest.mark.asyncio
async def test_ordered_apply_stream_refills_when_later_tasks_finish():
    docs = [Document(text=str(i)) for i in range(5)]
    f = ControlledFilter(count=5, batch_size=3)
    stream = f.apply_stream(docs)

    first_output = asyncio.create_task(stream.__anext__())
    for index in range(3):
        await asyncio.wait_for(f.started[index].wait(), timeout=1)

    # Items 1 and 2 finish before item 0. Their slots must be refilled even though
    # ordered output cannot yield anything until item 0 completes.
    f.release[1].set()
    await asyncio.wait_for(f.started[3].wait(), timeout=1)
    f.release[2].set()
    await asyncio.wait_for(f.started[4].wait(), timeout=1)

    assert not first_output.done()
    assert f.active == 3
    assert f.max_active == 3

    f.release[0].set()
    assert (await asyncio.wait_for(first_output, timeout=1)).text == "0"
    assert (await asyncio.wait_for(stream.__anext__(), timeout=1)).text == "1"
    assert (await asyncio.wait_for(stream.__anext__(), timeout=1)).text == "2"

    f.release[3].set()
    f.release[4].set()
    assert [doc.text async for doc in stream] == ["3", "4"]


@pytest.mark.asyncio
@pytest.mark.parametrize("ordered", [True, False])
async def test_apply_stream_does_not_wait_to_fill_window_from_async_source(ordered):
    release_second = asyncio.Event()

    async def source():
        yield Document("first")
        await release_second.wait()
        yield Document("second")

    f = UppercaseFilter(batch_size=2, ordered=ordered)
    stream = f.apply_stream(source())

    first = await asyncio.wait_for(stream.__anext__(), timeout=1)
    assert first.text == "FIRST"

    release_second.set()
    assert [doc.text async for doc in stream] == ["SECOND"]


@pytest.mark.asyncio
async def test_apply_stream_can_yield_in_completion_order():
    docs = [Document(text=str(i)) for i in range(2)]
    f = ControlledFilter(count=2, batch_size=2, ordered=False)
    stream = f.apply_stream(docs)

    first_output = asyncio.create_task(stream.__anext__())
    await asyncio.wait_for(f.started[0].wait(), timeout=1)
    await asyncio.wait_for(f.started[1].wait(), timeout=1)
    f.release[1].set()

    assert (await asyncio.wait_for(first_output, timeout=1)).text == "1"
    f.release[0].set()
    assert [doc.text async for doc in stream] == ["0"]


@pytest.mark.asyncio
async def test_apply_stream_cancels_pending_tasks_when_consumer_is_cancelled():
    docs = [Document(text=str(i)) for i in range(2)]
    f = ControlledFilter(count=2, batch_size=2)
    stream = f.apply_stream(docs)

    next_output = asyncio.create_task(stream.__anext__())
    await asyncio.wait_for(f.started[0].wait(), timeout=1)
    await asyncio.wait_for(f.started[1].wait(), timeout=1)

    next_output.cancel()
    with pytest.raises(asyncio.CancelledError):
        await next_output
    await stream.aclose()

    assert f.active == 0


@pytest.mark.asyncio
async def test_apply_stream_isolates_per_document_errors():
    class ErrorFilter(AsyncFilter):
        async def apply(self, document: Document) -> Document:
            if document.text == "bad":
                raise RuntimeError("fail")
            document.text = document.text.upper()
            return document

    docs = [Document(text="ok"), Document(text="bad"), Document(text="fine")]
    f = ErrorFilter(batch_size=3)
    outputs = [doc async for doc in f.apply_stream(docs)]

    assert [doc.text for doc in outputs] == ["OK", "bad", "FINE"]
    assert [doc.is_rejected for doc in outputs] == [False, True, False]
    assert "RuntimeError('fail')" in outputs[1].reject_reason["error"]
    assert f.get_statistics().errors == 1


@pytest.mark.asyncio
async def test_overridden_apply_batch_keeps_batch_processing():
    class BatchFilter(AsyncFilter):
        def __init__(self):
            super().__init__(batch_size=2)
            self.batch_sizes = []

        async def apply(self, document: Document) -> Document:
            raise AssertionError("apply should not be called")

        async def apply_batch(self, batch):
            self.batch_sizes.append(len(batch))
            return list(batch)

    f = BatchFilter()
    outputs = [doc async for doc in f.apply_stream([Document("a"), Document("b"), Document("c")])]
    assert [doc.text for doc in outputs] == ["a", "b", "c"]
    assert f.batch_sizes == [2, 1]


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        UppercaseFilter(batch_size=0)
