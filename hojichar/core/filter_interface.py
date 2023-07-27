import logging
from typing import Any, Dict, Optional, Set

from hojichar.core.models import Document, Token


def _is_jsonable(data: Any) -> bool:
    if data is None:
        return True
    elif isinstance(data, (bool, int, float, str)):
        return True
    """
    elif isinstance(data, (tuple, list)):
        return all(Filter._is_jsonable(x) for x in data)
    elif isinstance(data, dict):
        return all(isinstance(k, str) and Filter._is_jsonable(v) for k, v in data.items())
    """
    return False


class Filter:
    """
    Base class for all filters.
    Document-level filters must inherit from this class.

    The definition of filter function is in `apply` method.
    If you define a new filter, you must define the method.
    When this class is called, apply the filter from string to string.

    If the filter create `Document.tokens` form `Document.text`, you
    must implement `tokenize` method.
    If the filter update `Document.text` by merging `Document.tokens`, you
    must implement `merge` method.
    Do not define a filter that changes both `Document.text` and `Document.token`
    to prevent unexpected behavior.

    If you apply the filter to tokens, you can use `TokenFilter` class.

    Parameters
    ----------
    p: float
        The probability apply the filter organized by hojichar.Compose
    skip_reject: bool
        If set `True`, `hojichar.Compose` make this filter ignore the document
        which has `is_rejected` flag.
        This flag is `True` by default since processing discarded documents
        in subsequent filters is meaningless. However, in some cases, docs that
        have been rejected need another filter. For example, analyzing false-positive,
        discarded docs must be passed to JSON Dump filters. In such case,
        set the `skip_reject` flag as `False` and make it pass all docs.
    """

    def __init__(
        self, p: float = 1, skip_rejected: bool = True, *args: Any, **kwargs: Any
    ) -> None:
        """
        Parameters
        ----------
        p : float, opt∏ional
            Probability that this filter will be applied. Default=1
        """
        self.name = self.__class__.__name__
        self.logger = logging.getLogger("hojichar.document_filters." + self.name)
        assert 0 <= p <= 1
        self.p = p
        self.skip_rejected = skip_rejected

    def apply(self, document: Document) -> Document:
        """Definition of filter behavior.

        In this method, the filter will modify `document.text`, or
        set `document.is_rejected = True` to discard the document.

        Do not define a filter that changes both `document.text` and `document.token`

        Parameters
        ----------
        document : Document
            Input document

        Returns
        -------
        Document
            Processed Document
        """
        raise NotImplementedError(f"{self.__class__.__name__}.apply method is not defined")
        return document

    def apply_filter(self, document: Document) -> Document:
        document = self.apply(document)
        return document

    def __call__(self, text: str) -> str:
        document = Document(text)
        document = self.apply(document)
        return document.text

    def get_jsonalbe_vars(self, exclude_keys: Optional[Set[str]] = None) -> Dict[str, Any]:
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

    def get_jsonable_vars(self) -> dict:
        # Output key-values of member variables that can be obtained by var(self), except "logger".
        exclude_keys = ["logger"]
        return dict(filter(lambda item: item[0] not in exclude_keys, vars(self).items()))

    def get_jsonalbe_vars(self, exclude_keys: Optional[Set[str]] = None) -> dict:
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
