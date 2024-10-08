"""
.. include:: ../README.md
"""
from .core.composition import Compose
from .core.filter_interface import Filter, TokenFilter
from .core.inspection import StatsContainer
from .core.models import Document, Token
from .core.parallel import Parallel
from .filters import (
    deduplication,
    document_filters,
    language_identification,
    token_filters,
    tokenization,
)

__version__ = "0.0.0"  # Replaced by poetry-dynamic-versioning when deploying

__all__ = [
    "core",
    "filters",
    "utils",
    "Compose",
    "Filter",
    "TokenFilter",
    "Document",
    "Token",
    "Parallel",
    "StatsContainer",
    "deduplication",
    "document_filters",
    "language_identification",
    "token_filters",
    "tokenization",
]
