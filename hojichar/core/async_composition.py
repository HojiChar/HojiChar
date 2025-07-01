from __future__ import annotations

import asyncio
import logging
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Any, AsyncGenerator, AsyncIterable, Iterable, Sequence

import numpy as np

from hojichar.core.async_filter_interface import AsyncFilter
from hojichar.core.composition import Compose
from hojichar.core.filter_interface import Filter
from hojichar.core.models import Document, Statistics, get_doc_info
from hojichar.utils.async_handlers import handle_stream_as_async


class AsyncFilterAdapter(AsyncFilter):
    """
    Adapter class for executing hojichar.Filter asynchronously.
    """

    def __init__(
        self,
        sync_filter: Filter,
        *args: Any,
        executor: Executor | None = None,
        use_batch: bool = True,
        **kwargs: Any,
    ):
        """
        Adapter class for executing hojichar.Filter asynchronously.
        Used to incorporate Filter into AsyncCompose.

        To reduce the overhead of asynchronous context switching,
        use_batch is set to True by default to process in batches
        in apply_stream, regardless of the sync_filter's use_batch setting.

        If performing CPU-bound and heavy processing, you can specify an executor
        to offload the processing to the executor. However, due to Python's GIL
        constraints, using ThreadPoolExecutor will not parallelize CPU-bound
        processing, and the entire process will be locked.

        By using ProcessPoolExecutor as the executor, it may be possible to
        parallelize CPU-bound processing. However, for parallelizing CPU-bound
        processing, it is recommended to use the hojichar.Parallel class to
        parallelize synchronous Compose pipeline.
        """
        super().__init__(*args, use_batch=use_batch, **kwargs)
        self.sync_filter = sync_filter
        self._has_external_executor = executor is not None
        self._executor = executor or ThreadPoolExecutor()
        self.batch_size = sync_filter.batch_size

    async def apply(self, document: Document) -> Document:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.sync_filter.apply, document)

    async def apply_batch(self, batch: Sequence[Document]) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.sync_filter.apply_batch(batch),
        )

    async def shutdown(self) -> None:
        self.sync_filter.shutdown()
        if not self._has_external_executor:
            self._executor.shutdown()


class AsyncCompose(AsyncFilter):
    def __init__(
        self,
        filters: list[AsyncFilter | Filter],
        random_state: int | np.random.Generator | None = None,
        executor: ThreadPoolExecutor | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(random_state=random_state, *args, **kwargs)
        self.logger = logging.getLogger(f"{self.__module__}.{self.__class__.__name__}")
        self._statistics.name = "Total"
        self._has_external_executor = executor is not None
        self._executor = executor or ThreadPoolExecutor()
        self.set_filters(filters)

    def set_filters(self, filters: list[AsyncFilter | Filter]) -> None:
        self.filters: list[AsyncFilter] = []
        filter_idx = 0
        for f in filters:
            if isinstance(f, (AsyncCompose, Compose)):
                for sub in f.filters:
                    name = f"{filter_idx}-{sub.__class__.__name__}"
                    if isinstance(sub, Filter):
                        name = f"{filter_idx}-{sub.__class__.__name__}"
                        sub = AsyncFilterAdapter(sub, executor=self._executor)

                    sub._set_rng_if_not_initialized(self._rng)
                    sub.name = name
                    sub._statistics.name = name
                    self.filters.append(sub)
                    filter_idx += 1
            else:
                name = f"{filter_idx}-{f.__class__.__name__}"
                if isinstance(f, Filter):
                    name = f"{filter_idx}-{f.__class__.__name__}"
                    f = AsyncFilterAdapter(f, executor=self._executor)
                f._set_rng_if_not_initialized(self._rng)
                f.name = name
                f._statistics.name = name
                self.filters.append(f)
                filter_idx += 1

    async def apply(self, document: Document) -> Document:
        stat = get_doc_info(document)
        for filter_idx, filt in enumerate(self.filters):
            document = await filt._apply(document)
        new_stat = get_doc_info(document)
        async with self._stats_lock:
            self._statistics.update_by_diff(stat, new_stat)
        return document

    async def apply_batch(self, batch: Sequence[Document]) -> list[Document]:
        stats = [get_doc_info(doc) for doc in batch]
        for i, filt in enumerate(self.filters):
            batch = await filt._apply_batch(batch)
        batch = await self._finalize_batch(batch, stats)
        return list(batch)

    async def apply_stream(
        self,
        stream: AsyncIterable[Document] | Iterable[Document],
    ) -> AsyncGenerator[Document, None]:
        async_stream = handle_stream_as_async(stream, chunk_size=1000, executor=self._executor)
        async_stream = self._count_input_stats(async_stream)

        for i, filt in enumerate(self.filters):
            async_stream = filt.apply_stream(async_stream)

        async for doc in async_stream:
            in_stat = doc.extras["__init_stats"]
            out_stat = get_doc_info(doc)
            async with self._stats_lock:
                self._statistics.update_by_diff(in_stat, out_stat)
            del doc.extras["__init_stats"]
            yield doc

    async def _count_input_stats(
        self, async_stream: AsyncIterable[Document]
    ) -> AsyncGenerator[Document, None]:
        async for doc in async_stream:
            doc.extras["__init_stats"] = get_doc_info(doc)
            yield doc

    def get_total_statistics(self) -> list[Statistics]:
        """
        Get the statistics of the Compose object and sub filters.

        The statistics of the Compose class are stored in an object with the name "Total",
        and sub-filters's are stored with names in the format {filter_index}-{filter class name}.
        """
        stats = []
        stats.append(self.get_statistics())
        for i, filt in enumerate(self.filters):
            stats.append(filt.get_statistics())
        return stats

    def get_total_statistics_map(self) -> list[dict[str, Any]]:
        """
        Get the statistics of the Compose object and sub filters as a list of dictionaries.
        """
        stats = self.get_total_statistics()
        return [stat.to_dict() for stat in stats]

    async def shutdown(self) -> None:
        for filt in self.filters:
            await filt.shutdown()
        if not self._has_external_executor:
            self._executor.shutdown()

    async def __aenter__(self) -> "AsyncCompose":
        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        await self.shutdown()
        if exc_type is not None:
            raise exc_value
