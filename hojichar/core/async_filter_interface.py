from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterable,
    Awaitable,
    Callable,
    Iterable,
    Sequence,
    TypeVar,
)

import numpy as np

from hojichar.core.models import Document, Statistics, get_doc_info
from hojichar.utils.async_handlers import handle_stream_as_async

T = TypeVar("T")


def _is_jsonable(data: Any) -> bool:
    if data is None:
        return True
    elif isinstance(data, (bool, int, float, str)):
        return True
    return False


class AsyncFilter(ABC):
    def __init__(
        self,
        *args: Any,
        p: float = 1.0,
        skip_rejected: bool = True,
        random_state: int | np.random.Generator | None = None,
        use_batch: bool = True,
        batch_size: int = 128,
        **kwargs: Any,
    ):
        """
        Base class for asynchronous filters.

        Parameters
        ----------
        p : float
            The probability of applying the filter.
            If `p` is 1, the filter will always be applied.
        skip_rejected : bool
            If `True`, the filter will skip documents that are already rejected.
            If you want to apply the filter to all documents (e.g., postprocess), set this to `False`.
        random_state : Optional[Union[int, np.random.Generator]]
            Seed for the random number generator.
            If `None` is specified, the random number generator managed by the Compose class will be used.
        use_batch : bool
            If `True`, the filter will process documents in batches in the `apply_stream` method.
        batch_size : int
            The size of the batch to process documents in the `apply_stream` method.
        """
        self.name = self.__class__.__name__
        self.logger = logging.getLogger(f"{self.__module__}.{self.__class__.__name__}")
        assert 0 <= p <= 1
        self.p = p
        self.__init_rng(random_state)
        self.skip_rejected = skip_rejected
        self.use_batch = use_batch
        self.batch_size = batch_size

        self._statistics = Statistics(name=self.name)
        self._stats_lock = asyncio.Lock()

    @abstractmethod
    async def apply(self, document: Document) -> Document:
        """
        Definition of async filter behavior.

        In this method, the filter will modify `document.text` or
        `document.extras` and set `document.is_rejected = True` to discard the document.

        Parameters
        ----------
        document : Document
            Input document

        Returns
        -------
        Document
            Processed Document
        """
        pass

    def _check_skip(self, document: Document) -> bool:
        """
        Check if the document should be skipped by this filter.
        If `skip_rejected` is set to `True`, this method will return `True`
        if the document is already rejected.
        If `p` is less than 1, this method will return `True` with a probability of `1 - p`.
        """
        skip = self.skip_rejected and document.is_rejected
        if skip:
            return True
        if self.p < 1:
            if self._rng.random() > self.p:
                return True
        return False

    async def _apply(self, document: Document) -> Document:
        stats = get_doc_info(document)
        if not self._check_skip(document):
            document = await self.apply(document)
        new_stats = get_doc_info(document)
        async with self._stats_lock:
            self._statistics.update_by_diff(stats, new_stats)

        if not stats["is_rejected"] and new_stats["is_rejected"]:
            document.reject_reason = self.get_jsonable_vars()
        return document

    async def apply_batch(self, batch: Sequence[Document]) -> list[Document]:
        """
        Apply the filter to a Sequence of documents.
        By default, the processing implemented in `apply` is executed asynchronously and concurrently.
        If the filter processing can be optimized for batch processing, override this method.
        """
        tasks = [self.apply(doc) for doc in batch]
        return await asyncio.gather(*tasks)

    async def _apply_batch(self, batch: Sequence[Document]) -> list[Document]:
        skip = False
        if self.p < 1:
            skip = self._rng.random() > self.p

        stats = [get_doc_info(doc) for doc in batch]
        if not skip:
            batch = await self.apply_batch(batch)
        batch = await self._finalize_batch(batch, stats)
        return list(batch)

    async def apply_stream(
        self,
        stream: Iterable[Document] | AsyncIterable[Document],
    ) -> AsyncGenerator[Document, None]:
        """
        Apply the filter to a stream of documents (Iterable or AsyncIterable).
        If use_batch is set to `True` at initialization, the filter will process documents in batches.
        If the stream is not asynchronous, use handle_stream_as_async to convert it to an asynchronous stream.

        Even if an exception occurs during processing, the process will continue, and the following actions will be taken:
        - Set the `is_rejected` flag of the document to `True`
        - Set the error details in `reject_reason`
        - Increment the `errors` count in the statistics retrievable via `get_statistics`
        """
        async_stream: AsyncIterable[Document] = handle_stream_as_async(stream)

        if not self.use_batch:
            async for doc in async_stream:
                yield await self._try_process(doc, self._apply)
        else:
            batch: list[Document] = []
            async for doc in async_stream:
                if self._check_skip(doc):
                    yield doc
                    continue

                batch.append(doc)
                # Batch size reached, apply batch
                if len(batch) >= self.batch_size:
                    stats = [get_doc_info(doc) for doc in batch]
                    batch = await self._try_process(batch, self.apply_batch)
                    batch = await self._finalize_batch(batch, stats)
                    for out in batch:
                        yield out
                    batch.clear()

            # Flush remaining documents in the batch
            if batch:
                stats = [get_doc_info(doc) for doc in batch]
                batch = await self._try_process(batch, self.apply_batch)
                batch = await self._finalize_batch(batch, stats)
                for out in batch:
                    yield out

    async def _try_process(self, target: T, func: Callable[[T], Awaitable[T]]) -> T:
        try:
            return await func(target)
        except Exception as e:
            if isinstance(target, Document):
                msg = f"{e!r} occurs while processing {self.name} with {target!r}"
                self.logger.error(msg, exc_info=True)
                target.is_rejected = True
                target.reject_reason = {"error": msg}
                async with self._stats_lock:
                    self._statistics.errors += 1
                return target  # type: ignore[return-value]
            if isinstance(target, list):
                msg = f"{e!r} occurs while batch processing {self.name}"
                self.logger.error(msg, exc_info=True)
                for doc in target:
                    doc.is_rejected = True
                    doc.reject_reason = {"error": msg}
                async with self._stats_lock:
                    self._statistics.errors += len(target)
                return target  # type: ignore[return-value]
            raise e

    async def __call__(self, text: str) -> str:
        document = Document(text=text)
        return (await self._apply(document)).text

    def get_statistics(self) -> Statistics:
        """
        Get the statistics of this filter.
        Returns:
            Statistics: The statistics of this filter.
        """
        return self._statistics

    def get_statistics_map(self) -> dict[str, Statistics]:
        """
        Get the statistics of this filter as a dictionary.
        """
        return self._statistics.to_dict()

    async def shutdown(self) -> None:
        """
        You can override this method to release resources or perform cleanup tasks.
        """
        pass

    async def __aenter__(self) -> "AsyncFilter":
        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        await self.shutdown()

    def get_jsonable_vars(self, exclude_keys: set[str] | None = None) -> dict[str, Any]:
        """
        Get the member variable of this filter.
        Eligible variables are primitive types; [bool, int, float, str, None],
        and the name of the variable not starts with the underscore; `_`.
        """
        if exclude_keys is None:
            exclude_keys = set()
        return {
            k: v
            for k, v in vars(self).items()
            if (_is_jsonable(v) and (k not in exclude_keys) and (not k.startswith("_")))
        }

    async def _finalize_batch(
        self,
        batch: Sequence[Document],
        old_stats: list[dict[str, Any]],
    ) -> list[Document]:
        new_stats = [get_doc_info(doc) for doc in batch]
        for old, new, doc in zip(old_stats, new_stats, batch):
            async with self._stats_lock:
                self._statistics.update_by_diff(old, new)
            if not old["is_rejected"] and new["is_rejected"]:
                doc.reject_reason = self.get_jsonable_vars()
        return list(batch)

    def __init_rng(self, random_state: int | np.random.Generator | None) -> None:
        self._owns_rng = True
        if random_state is None:
            self._rng = np.random.default_rng()
            self._owns_rng = False
        elif isinstance(random_state, int):
            self._rng = np.random.default_rng(random_state)
        elif isinstance(random_state, np.random.Generator):
            self._rng = random_state
        else:
            raise TypeError(
                f"random_state must be int or np.random.Generator, not {type(random_state)}"
            )

    def _set_rng_if_not_initialized(self, rng: np.random.Generator) -> None:
        """
        Set the random number generator for this filter if it is not already initialized.
        This method is called by Compose class.
        """
        if not self._owns_rng:
            self._rng = rng
