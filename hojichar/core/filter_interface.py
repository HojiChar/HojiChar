import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, TypeVar, Union

import numpy as np

from hojichar.core.models import Document, Statistics, Token, get_doc_info
from hojichar.utils.warn_deprecation import deprecated_since

T = TypeVar("T")


def _is_jsonable(data: Any) -> bool:
    if data is None:
        return True
    elif isinstance(data, (bool, int, float, str)):
        return True
    return False


class Filter(ABC):
    """
    Base class for all filters.
    Document-level filters must inherit from this class.

    The definition of text processing is in `apply` method.
    If you define a new filter, override the method.

    When this class is called, apply the filter from string to string.

    With context manager, you can use the filter as follows:
    ```python
    with YourFilter(p=0.5) as filt:
        text = filt("This is a sample text.")
    ```

    """

    def __init__(
        self,
        p: float = 1.0,
        skip_rejected: bool = True,
        *args: Any,
        random_state: Optional[Union[int, np.random.Generator]] = None,
        use_batch: bool = False,
        batch_size: int = 128,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the filter.
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
            If `None`, a new random number generator will be created.
            If `None`, and use in the `Compose` class, the random state is shared with the `Compose` object.
        use_batch : bool
            If `True`, the filter will process documents in batches in the `apply_stream` method.
        batch_size : int
            The size of the batch to process documents in the `apply_stream` method.
        kwargs : Any
            Additional keyword arguments to pass to the filter.
        """
        self.name = self.__class__.__name__
        self.logger = logging.getLogger(f"{self.__module__}.{self.__class__.__name__}")
        assert 0 <= p <= 1
        self.p = p
        self.__init_rng(random_state)
        self.skip_rejected = skip_rejected
        self.use_batch = use_batch
        self.batch_size = batch_size

        self._statistics: Statistics = Statistics()

    @abstractmethod
    def apply(self, document: Document) -> Document:
        """
        Definition of filter behavior.

        The document must have a protocol `TextContent`,
        and mostly used hojichar.Document class.

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

    @deprecated_since(version="1.0.0", alternative="apply")
    def apply_filter(self, document: Document) -> Document:
        document = self.apply(document)
        return document

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

    def _apply(self, document: Document) -> Document:
        """
        Apply the filter to a single document.
        This method
          - checks if the document should be skipped
          - counts and logging the statistics
          - logging the reason for rejection if the document is rejected

        This method may be used in `apply` method of `Compose` class.
        """

        stats = get_doc_info(document)

        if not self._check_skip(document):
            document = self.apply(document)

        new_stats = get_doc_info(document)
        self._statistics.update_by_diff(stats, new_stats)

        if not stats["is_rejected"] and new_stats["is_rejected"]:
            document.reject_reason = self.get_jsonable_vars()

        return document

    def apply_batch(self, batch: Sequence[Document]) -> List[Document]:
        """
        Apply the filter to a batch of documents.
        You can override this method if you want to
        apply the filter to a batch of documents at once.

        This method may be used in `apply_batch` method of `Compose` class.

        Parameters
        ----------
        documents : Sequence[Document]
            List-like object of input documents

        Returns
        -------
        list[Document]
            List of processed documents
        """
        return [self.apply(document) for document in batch]

    def _apply_batch(self, batch: Sequence[Document]) -> List[Document]:
        """
        Apply the filter to a batch of documents.
        This method
        - checks if the documents should be skipped
        - counts and logs the statistics
        - logs the reason for rejection if any document is rejected
        """
        skip = False
        if self.p < 1:
            skip = self._rng.random() > self.p

        stats = [get_doc_info(document=doc) for doc in batch]
        if not skip:
            batch = self.apply_batch(batch)
        batch = self._finalize_batch(batch, stats)
        return batch

    def apply_stream(self, stream: Iterable[Document]) -> Iterable[Document]:
        """
        Apply the filter to a stream of documents.
        This method is used when you want to process documents one by one.
        If `use_batch` is set to `True` in the constructor,
        this method will process documents in batches using the `apply_batch` method.

        Even if an exception occurs during processing, the process will continue, and the following actions will be taken:
        - Set the `is_rejected` flag of the document to `True`
        - Set the error details in `reject_reason`
        - Increment the `errors` count in the statistics retrievable via `get_statistics`

        Parameters
        ----------
        stream : Iterable[Document]
            Stream of input documents

        Returns
        -------
        Iterable[Document]
            Stream of processed documents
        """

        if not self.use_batch:
            for document in stream:
                yield self._try_process(document, self._apply)
        else:
            batch: list[Document] = []
            for document in stream:
                if self._check_skip(document):
                    yield document
                    continue

                batch.append(document)
                if len(batch) >= self.batch_size:
                    stats = [get_doc_info(doc) for doc in batch]
                    batch = self._try_process(batch, self.apply_batch)
                    batch = self._finalize_batch(batch, stats)
                    yield from batch
                    batch.clear()
            if batch:
                stats = [get_doc_info(doc) for doc in batch]
                batch = self._try_process(batch, self.apply_batch)
                batch = self._finalize_batch(batch, stats)
                yield from batch

    def _try_process(self, target: T, func: Callable[[T], T]) -> T:
        try:
            return func(target)
        except Exception as e:
            if isinstance(target, Document):
                msg = f"{e!r} occurs while processing {self.name} with {target!r}"
                target.is_rejected = True
                target.reject_reason = {"error": msg}
                self._statistics.errors += 1
                self.logger.error(msg, exc_info=True)
                return target  # type: ignore[return-value]
            if isinstance(target, list):
                msg = f"{e!r} occurs while batch processing {self.name}"
                self.logger.error(msg, exc_info=True)
                for doc in target:
                    doc.is_rejected = True
                    doc.reject_reason = {"error": msg}
                self._statistics.errors += len(target)
                return target  # type: ignore[return-value]
            else:
                raise e

    def __call__(self, text: str, **kwargs: Any) -> str:
        document = Document(text, **kwargs)
        document = self._apply(document)
        return document.text

    def get_statistics(self) -> Statistics:
        """
        Get the statistics of this filter.
        This method returns the statistics of the filter,
        which includes the number of processed documents, discarded documents, and other statistics.
        """
        return self._statistics

    def get_statistics_map(self) -> Dict[str, Any]:
        """
        Get the statistics of this filter as a dictionary.
        """
        return self._statistics.to_dict()

    def shutdown(self) -> None:
        """
        This method is called when the filter is no longer needed.
        You can override this method to release resources or perform cleanup tasks.
        """
        pass

    def __enter__(self) -> "Filter":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """
        This method is called when the filter is used in a context manager.
        It calls the `shutdown` method to release resources or perform cleanup tasks.
        """
        self.shutdown()

    def get_jsonable_vars(self, exclude_keys: Optional[Set[str]] = None) -> Dict[str, Any]:
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

    def _finalize_batch(
        self: "Filter",
        batch: Sequence[Document],
        old_stats: List[Dict[str, Any]] = [],
    ) -> List[Document]:
        new_stats = [get_doc_info(doc) for doc in batch]
        for old, new, doc in zip(old_stats, new_stats, batch):
            self._statistics.update_by_diff(old, new)
            if not old["is_rejected"] and new["is_rejected"]:
                doc.reject_reason = self.get_jsonable_vars()
        return list(batch)

    def __init_rng(self, random_state: Optional[Union[int, np.random.Generator]]) -> None:
        self._owns_rng = True
        if random_state is None:
            self._rng = np.random.default_rng()
            self._owns_rng = False
        elif isinstance(random_state, int):
            self._rng = np.random.default_rng(random_state)
        elif isinstance(random_state, np.random.Generator):
            self._rng = random_state

    def _set_rng_if_not_initialized(self, rng: np.random.Generator) -> None:
        """
        Set the random number generator for this filter if it is not already initialized.
        This method is called by Compose class.
        """
        if not self._owns_rng:
            self._rng = rng


@deprecated_since(version="1.0.0", alternative="Filter")
class TokenFilter(Filter, ABC):
    """
    Base class for token-level filters.

    Token filters, which shuld be implemented in hojichar/filters/token_filters.py,
    must inherit from this class.
    """

    def __init__(
        self, p: float = 1, skip_rejected: bool = True, *args: Any, **kwargs: Any
    ) -> None:
        self.name = self.__class__.__name__
        self.logger = logging.getLogger("hojichar.token_filters." + self.name)
        assert 0 <= p <= 1
        self.p = p
        self.skip_rejected = skip_rejected

    def apply(self, token: Token) -> Token:  # type: ignore
        raise NotImplementedError(f"{self.__class__.__name__}.apply method is not defined")
        return token

    def apply_filter(self, document: Document) -> Document:
        document.tokens = [self.apply(token) for token in document.tokens if not token.is_rejected]
        return document

    def __call__(self, text: str) -> str:  # type: ignore
        token = Token(text)
        token = self.apply(token)
        return token.text

    def _apply(self, document: Document) -> Document:
        """
        Apply the token filter to a single document.
        This method checks if the document should be skipped.
        """
        if self.skip_rejected and document.is_rejected:
            return document
        return self.apply_filter(document)

    def get_jsonable_vars(self, exclude_keys: Optional[Set[str]] = None) -> dict:
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
