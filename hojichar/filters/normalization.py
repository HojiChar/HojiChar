import logging
import unicodedata

from hojichar.core.filter_interface import Filter
from hojichar.core.models import Document


class DocumentNormalizer(Filter):
    """
    正規化をします.
    """

    def apply(self, document: Document) -> Document:
        document.text = unicodedata.normalize("NFKC", document.text)
        return document


if __name__ == "__main__":
    import doctest

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s]%(name)s:%(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.DEBUG)

    doctest.testmod()
