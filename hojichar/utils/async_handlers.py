from __future__ import annotations

import asyncio
import itertools
import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterable,
    Awaitable,
    Callable,
    Generic,
    Iterable,
    Iterator,
    TextIO,
    TypeVar,
    cast,
)

T = TypeVar("T")
logger = logging.getLogger(__name__)


class _AsyncIteratorError:
    def __init__(self, error: BaseException):
        self.error = error


_ASYNC_ITERATOR_END = object()


class AsyncToSyncIterator(Iterator[T], Generic[T]):
    """Consume an async iterable from synchronous code.

    A single background thread owns the event loop for the lifetime of the iterator. Use
    this class as a context manager when iteration may stop before the source is exhausted.
    """

    def __init__(
        self,
        source_stream: AsyncIterable[T],
        *,
        buffer_size: int = 128,
        finalizer: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        if buffer_size < 1:
            raise ValueError("buffer_size must be at least 1")

        self._source_stream = source_stream
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=buffer_size)
        self._finalizer = finalizer
        self._stop_event = threading.Event()
        self._started_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._consumer_task: asyncio.Task[None] | None = None
        self._closed = False

    def __iter__(self) -> "AsyncToSyncIterator[T]":
        return self

    def __next__(self) -> T:
        if self._closed:
            raise StopIteration
        self._start()

        item = self._queue.get()
        if item is _ASYNC_ITERATOR_END:
            self.close()
            raise StopIteration
        if isinstance(item, _AsyncIteratorError):
            self.close()
            raise item.error
        return cast(T, item)

    def __enter__(self) -> "AsyncToSyncIterator[T]":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        thread_was_started = self._thread is not None

        if not thread_was_started:
            # Closing an unconsumed iterator must still close its source and run its finalizer.
            self._thread = threading.Thread(
                target=self._thread_main,
                name="hojichar-async-to-sync",
                daemon=True,
            )
            self._thread.start()
            self._started_event.wait()

        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        if (
            thread_was_started
            and self._thread is not None
            and self._thread.is_alive()
            and self._loop is not None
            and not self._loop.is_closed()
            and self._consumer_task is not None
        ):
            try:
                self._loop.call_soon_threadsafe(self._consumer_task.cancel)
            except RuntimeError:
                # The background loop can close between is_closed() and scheduling.
                pass

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("Async-to-sync iterator did not stop within 5 seconds")

    def _start(self) -> None:
        if self._thread is not None:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "Cannot synchronously consume an async iterable from a running event loop"
            )

        self._thread = threading.Thread(
            target=self._thread_main,
            name="hojichar-async-to-sync",
            daemon=True,
        )
        self._thread.start()
        self._started_event.wait()

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._consume())
        except BaseException as error:
            if not self._stop_event.is_set():
                self._put(_AsyncIteratorError(error))
            elif not isinstance(error, asyncio.CancelledError):
                logger.error("Failed to close async-to-sync iterator", exc_info=True)
        finally:
            self._started_event.set()
            if not self._stop_event.is_set():
                self._put(_ASYNC_ITERATOR_END)

    async def _consume(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._consumer_task = asyncio.current_task()
        self._started_event.set()
        iterator = None

        try:
            iterator = self._source_stream.__aiter__()
            while not self._stop_event.is_set():
                try:
                    item = await iterator.__anext__()
                except StopAsyncIteration:
                    break
                if not self._put(item):
                    break
        finally:
            try:
                aclose = getattr(iterator, "aclose", None)
                if aclose is not None:
                    await aclose()
            finally:
                if self._finalizer is not None:
                    await self._finalizer()

    def _put(self, item: Any) -> bool:
        while not self._stop_event.is_set():
            try:
                self._queue.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False


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


def handle_async_stream_as_sync(
    source_stream: AsyncIterable[T],
    *,
    buffer_size: int = 128,
    finalizer: Callable[[], Awaitable[None]] | None = None,
) -> AsyncToSyncIterator[T]:
    """Convert an async iterable to a synchronous, closeable iterator.

    The iterator owns a background event-loop thread. Fully consuming it closes the source
    automatically. Use it as a context manager when the consumer may stop early.
    """
    return AsyncToSyncIterator(
        source_stream,
        buffer_size=buffer_size,
        finalizer=finalizer,
    )


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


async def fileout_from_async_iter(
    fp: TextIO, iter: AsyncIterable[str], buffer_size: int = 128
) -> None:
    buffer = []
    async for line in iter:
        buffer.append(line + "\n")
        if len(buffer) >= buffer_size:
            await asyncio.to_thread(fp.write, "".join(buffer))
            buffer.clear()
    await asyncio.to_thread(fp.writelines, buffer)
    buffer.clear()
