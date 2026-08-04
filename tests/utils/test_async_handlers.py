import asyncio
import threading
from typing import AsyncIterable

import pytest

from hojichar.utils.async_handlers import handle_async_stream_as_sync, handle_stream_as_async


@pytest.mark.asyncio
async def test_sync_iterable_basic():
    data = [1, 2, 3, 4, 5]
    agen = handle_stream_as_async(data, chunk_size=2)
    result = [x async for x in agen]
    assert result == data


@pytest.mark.asyncio
async def test_sync_iterable_empty():
    data = []
    agen = handle_stream_as_async(data, chunk_size=3)
    result = [x async for x in agen]
    assert result == []


@pytest.mark.asyncio
async def test_sync_iterable_chunk_boundaries():
    data = list(range(7))
    agen = handle_stream_as_async(data, chunk_size=3)
    # Expect chunks: [0,1,2], [3,4,5], [6]
    result = [x async for x in agen]
    assert result == data


@pytest.mark.asyncio
async def test_sync_iterable_chunk_size_one():
    data = list(range(5))
    agen = handle_stream_as_async(data, chunk_size=1)
    result = [x async for x in agen]
    assert result == data


@pytest.mark.asyncio
async def test_async_iterable_passthrough():
    async def source():
        for i in range(3):
            yield i

    agen_source = source()
    assert isinstance(agen_source, AsyncIterable)

    # Passthrough: should return the same object
    agen = handle_stream_as_async(agen_source, chunk_size=1)
    assert agen is agen_source
    result = [x async for x in agen]
    assert result == [0, 1, 2]


@pytest.mark.asyncio
async def test_large_chunk_size_exceeds_length():
    data = [10, 20, 30]
    agen = handle_stream_as_async(data, chunk_size=100)
    result = [x async for x in agen]
    assert result == data


def test_async_stream_as_sync_basic():
    async def source():
        for i in range(5):
            yield i

    with handle_async_stream_as_sync(source(), buffer_size=2) as stream:
        assert list(stream) == [0, 1, 2, 3, 4]


def test_async_stream_as_sync_propagates_exceptions():
    async def source():
        yield 1
        raise ValueError("broken stream")

    with handle_async_stream_as_sync(source()) as stream:
        assert next(stream) == 1
        with pytest.raises(ValueError, match="broken stream"):
            next(stream)


def test_async_stream_as_sync_close_cancels_source():
    source_closed = threading.Event()

    async def source():
        try:
            yield 1
            await asyncio.Event().wait()
        finally:
            source_closed.set()

    with handle_async_stream_as_sync(source()) as stream:
        assert next(stream) == 1

    assert source_closed.wait(timeout=1)


def test_async_stream_as_sync_runs_finalizer():
    finalized = threading.Event()

    async def source():
        yield 1

    async def finalize():
        finalized.set()

    assert list(handle_async_stream_as_sync(source(), finalizer=finalize)) == [1]
    assert finalized.wait(timeout=1)


def test_async_stream_as_sync_close_before_consumption_runs_finalizer():
    finalized = threading.Event()

    async def source():
        yield 1

    async def finalize():
        finalized.set()

    stream = handle_async_stream_as_sync(source(), finalizer=finalize)
    stream.close()

    assert finalized.wait(timeout=1)


def test_async_stream_as_sync_validates_buffer_size():
    async def source():
        yield 1

    stream = source()
    with pytest.raises(ValueError, match="buffer_size must be at least 1"):
        handle_async_stream_as_sync(stream, buffer_size=0)
    asyncio.run(stream.aclose())


@pytest.mark.asyncio
async def test_async_stream_as_sync_rejects_running_event_loop():
    async def source():
        yield 1

    async_source = source()
    stream = handle_async_stream_as_sync(async_source)
    with pytest.raises(RuntimeError, match="running event loop"):
        next(stream)
    await async_source.aclose()
