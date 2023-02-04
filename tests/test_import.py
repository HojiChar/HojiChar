# flake8: noqa
import sys


def test_root_import():
    import hojichar


def test_alias_import():
    from hojichar import (
        Compose,
        Document,
        Filter,
        Token,
        TokenFilter,
        deduplication,
        document_filters,
        token_filters,
        tokenization,
    )


def test_long_import():
    from hojichar.core.composition import Compose
    from hojichar.core.filter_interface import Filter, TokenFilter
    from hojichar.core.models import Document, Token
    from hojichar.filters import deduplication, document_filters, token_filters, tokenization
