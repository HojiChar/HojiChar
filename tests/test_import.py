# flake8: noqa
import sys


def test_root_import():
    import hojichar


def test_alias_import():
    from hojichar import (
        Compose,
        Document,
        Inspector,
        Token,
        deduplication,
        document_filters,
        token_filters,
        tokenization,
    )


def test_long_impor():
    from hojichar.core.composition import Compose
    from hojichar.core.inspection import Inspector
    from hojichar.core.models import Document, Token
    from hojichar.filters import document_filters
