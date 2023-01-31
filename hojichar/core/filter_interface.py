import logging

from hojichar.core.models import Document, Token


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
    """

    def __init__(self, p: float = 1, *args, **kwargs):
        """
        Parameters
        ----------
        p : float, optional
            Probability that this filter will be applied. Default=1
        """
        self.name = self.__class__.__name__
        self.logger = logging.getLogger("hojichar.document_filters." + self.name)
        assert 0 <= p <= 1
        self.p = p

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

    def filter_apply(self, document: Document) -> Document:
        if document.is_rejected:
            return document
        else:
            return self.apply(document)

    def __call__(self, text: str) -> str:
        document = Document(text)
        document = self.filter_apply(document)
        return document.text


class TokenFilter:
    """
    Base class for token-level filters.

    Token filters, which shuld be implemented in hojichar/filters/token_filters.py,
    must inherit from this class.
    """

    def __init__(self, p=1, *args, **kwargs):
        self.name = self.__class__.__name__
        self.logger = logging.getLogger("hojichar.token_filters." + self.name)
        assert 0 <= p <= 1
        self.p = p

    def apply(self, token: Token) -> Token:
        raise NotImplementedError(f"{self.__class__.__name__}.apply method is not defined")
        return token

    def filter_apply(self, document: Document) -> Document:
        if document.is_rejected:
            return document
        else:
            document.tokens = [
                self.apply(token) for token in document.tokens if not token.is_rejected
            ]
            return document

    def __call__(self, text: str) -> str:
        token = Token(text)
        token = self.apply(token)
        return token.text
