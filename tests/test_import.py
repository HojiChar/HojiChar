# flake8: noqa
import sys


def test_root_import() -> None:
    import hojichar


def test_alias_import() -> None:
    from hojichar import (
        Compose,
        Document,
        Filter,
        Parallel,
        StatsContainer,
        Token,
        TokenFilter,
        deduplication,
        document_filters,
        token_filters,
        tokenization,
    )


def test_long_import() -> None:
    from hojichar.core.composition import Compose
    from hojichar.core.filter_interface import Filter, TokenFilter
    from hojichar.core.inspection import StatsContainer
    from hojichar.core.models import Document, Token
    from hojichar.core.parallel import Parallel
    from hojichar.filters import deduplication, document_filters, token_filters, tokenization
