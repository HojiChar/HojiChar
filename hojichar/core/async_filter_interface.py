from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, AsyncIterable, Iterable, Sequence

import numpy as np

from hojichar.core.models import Document
from hojichar.utils.async_handlers import handle_stream_as_async


class AsyncFilter(ABC):
    def __init__(
        self,
        *args: Any,
        p: float = 1.0,
        skip_rejected: bool = True,
        random_state: int | np.random.Generator | None = None,
        use_batch: bool = False,
        batch_size: int = 128,
        **kwargs: Any,
    ):
        """
        非同期フィルターの基底クラス

        :param p: フィルター適用確率 (デフォルトは 1.0)
        :param skip_rejected: True のとき、 Document.is_rejected が True であれば処理をスキップする. JSONDumper などの全テキストに
        適用するべき後処理の場合は False にする
        """
        self.name = self.__class__.__name__
        assert 0 <= p <= 1
        self.p = p
        self.__init_rng(random_state)
        self.skip_rejected = skip_rejected
        self._use_batch = use_batch
        self._batch_size = batch_size

    @abstractmethod
    async def apply(self, document: Document) -> Document:
        """
        Definition of filter behavior.

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
        if self._check_skip(document):
            return document
        return await self.apply(document)

    async def apply_batch(self, batch: Sequence[Document]) -> list[Document]:
        """
        ドキュメントのSequenceに対してフィルターを適用する
        デフォルトでは apply に実装した処理を非同期に同時実行する。
        フィルタ処理がバッチ処理が可能な場合はオーバーライドして効率化できる
        """
        tasks = [self._apply(doc) for doc in batch]
        return await asyncio.gather(*tasks)

    async def _apply_batch(self, batch: Sequence[Document]) -> list[Document]:
        skip = False
        if self.p < 1:
            skip = self._rng.random() > self.p

        if not skip:
            batch = await self.apply_batch(batch)

        return list(batch)

    async def apply_stream(
        self,
        stream: Iterable[Document] | AsyncIterable[Document],
    ) -> AsyncGenerator[Document, None]:
        """
        ドキュメントのストリーム(Iterable)に対してフィルターを適用する
        ドキュメントをバッチサイズごとにバッファリングし、apply_batch を呼び出す。
        ストリームが非同期でない場合は、handle_stream_as_async を使って非同期ストリームに変換する。
        """
        if isinstance(stream, AsyncIterable):
            async_stream = stream
        else:
            async_stream = handle_stream_as_async(stream, self._batch_size)

        if not self._use_batch:
            async for doc in async_stream:
                yield await self._apply(doc)
        else:
            batch: list[Document] = []
            async for doc in async_stream:
                if self._check_skip(doc):
                    yield doc
                    continue

                batch.append(doc)
                # Batch size reached, apply batch
                if len(batch) >= self._batch_size:
                    outputs: list[Document] = await self.apply_batch(batch)
                    for out in outputs:
                        yield out
                    batch.clear()

            # Flush remaining documents in the batch
            if batch:
                outputs = await self.apply_batch(batch)
                for out in outputs:
                    yield out

    async def __call__(self, text: str) -> str:
        document = Document(text=text)
        return (await self._apply(document)).text

    def shutdown(self) -> None:
        pass

    def __init_rng(self, random_state: int | np.random.Generator | None) -> None:
        self._owns_rng = True
        if random_state is None:
            self._rng = np.random.default_rng()
            self._owns_rng = False
        elif isinstance(random_state, int):
            self._rng = np.random.default_rng(random_state)
        elif isinstance(random_state, np.random.Generator):
            self._rng = random_state
