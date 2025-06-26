"""
.. include:: ../README.md
"""

from .core.async_composition import AsyncCompose, AsyncFilterAdapter
from .core.async_filter_interface import AsyncFilter
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

__version__ = "0.0.0"  # Replaced by uv-dynamic-versioning when deploying

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
    "AsyncCompose",
    "AsyncFilterAdapter",
    "AsyncFilter",
]
