from typing import AsyncIterable

import pytest

from hojichar.utils.async_handlers import handle_stream_as_async


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
