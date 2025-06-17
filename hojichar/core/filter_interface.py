import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional, Set, TypeVar

import numpy as np

from hojichar.core.models import Document, Token
from hojichar.utils.warn_deprecation import deprecated_since


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
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
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
        seed : Optional[int]
            Seed for the random number generator.
        rng : Optional[np.random.Generator]
            A numpy random number generator instance.
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
        self.__init_rng(seed=seed, rng=rng)
        self.skip_rejected = skip_rejected
        self._use_batch = use_batch
        self._batch_size = batch_size

    def __init_rng(
        self, seed: Optional[int] = None, rng: Optional[np.random.Generator] = None
    ) -> None:
        if seed is not None and rng is not None:
            raise ValueError("You cannot set both `seed` and `rng` at the same time.")

        self._give_rng_at_init = True
        if rng is not None:
            self._rng = rng
        elif seed is not None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()
            self._give_rng_at_init = False

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
        document : T
            Input document

        Returns
        -------
        T
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
        This method checks if the document should be skipped.
        """
        if self._check_skip(document):
            return document
        return self.apply(document)

    def apply_batch(self, documents: list[Document]) -> list[Document]:
        """
        Apply the filter to a batch of documents.
        You can override this method if you want to
        apply the filter to a batch of documents at once.

        Parameters
        ----------
        documents : list[Document]
            List of input documents

        Returns
        -------
        list[Document]
            List of processed documents
        """
        return [self._apply(document) for document in documents]

    def apply_stream(self, document_stream: Iterable[Document]) -> Iterable[Document]:
        """
        Apply the filter to a stream of documents.
        This method is used when you want to process documents one by one.
        If `use_batch` is set to `True` in the constructor,
        this method will process documents in batches.

        Parameters
        ----------
        document_stream : Iterable[Document]
            Stream of input documents

        Returns
        -------
        Iterable[Document]
            Stream of processed documents
        """

        if not self._use_batch:
            for document in document_stream:
                yield self._apply(document)
        else:
            batch: list[Document] = []
            for document in document_stream:
                if self._check_skip(document):
                    yield document
                    continue

                batch.append(document)
                if len(batch) >= self._batch_size:
                    yield from self.apply_batch(batch)
                    batch.clear()
            if batch:
                yield from self.apply_batch(batch)

    def __call__(self, text: str, **kwargs: Any) -> str:
        document = Document(text, **kwargs)
        document = self._apply(document)
        return document.text

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


@deprecated_since(version="1.0.0", alternative="Filter")
class TokenFilter:
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

    def apply(self, token: Token) -> Token:
        raise NotImplementedError(f"{self.__class__.__name__}.apply method is not defined")
        return token

    def apply_filter(self, document: Document) -> Document:
        document.tokens = [self.apply(token) for token in document.tokens if not token.is_rejected]
        return document

    def __call__(self, text: str) -> str:
        token = Token(text)
        token = self.apply(token)
        return token.text

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
