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
