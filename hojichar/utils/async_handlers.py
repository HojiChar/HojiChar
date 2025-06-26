from __future__ import annotations

import asyncio
import itertools
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncGenerator, AsyncIterable, Iterable, TypeVar

T = TypeVar("T")


def handle_stream_as_async(
    source_stream: Iterable[T] | AsyncIterable[T],
    chunk_size: int = 1000,
    executor: ThreadPoolExecutor | None = None,
) -> AsyncGenerator[T, None]:
    """
    Convert a synchronous iterable to an asynchronous generator
    with a specified chunk size.

    Args:
        source_stream (Iterable[T]): The synchronous iterable to convert.
        chunk_size (int): The number of items to yield at a time.
    """
    if isinstance(source_stream, AsyncIterable):
        return source_stream  # type: ignore[return-value]
    stream = iter(source_stream)

    async def sync_to_async() -> AsyncGenerator[T, None]:
        loop = asyncio.get_running_loop()
        while True:
            chunk = await loop.run_in_executor(
                executor, lambda: list(itertools.islice(stream, chunk_size))
            )
            if not chunk:
                break
            for item in chunk:
                yield item

    return sync_to_async()


async def write_stream_to_file(
    stream: AsyncGenerator[str, None],
    output_path: Path | str,
    *,
    chunk_size: int = 1000,
    delimiter: str = "\n",
) -> None:
    """
    Write an asynchronous stream of strings to a file.
    To lessen overhead with file I/O, it writes in chunks.
    """
    loop = asyncio.get_running_loop()
    with open(output_path, "w", encoding="utf-8") as f:
        chunk = []
        async for line in stream:
            chunk.append(line)
            if len(chunk) >= chunk_size:
                await loop.run_in_executor(None, f.writelines, [s + delimiter for s in chunk])
                chunk = []
        if chunk:
            await loop.run_in_executor(None, f.writelines, [s + delimiter for s in chunk])
            chunk = []
        await loop.run_in_executor(None, f.flush)
