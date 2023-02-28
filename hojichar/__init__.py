"""
.. include:: ../README.md
"""
from .core.composition import Compose
from .core.filter_interface import Filter, TokenFilter
from .core.models import Document, Token
from .filters import deduplication, document_filters, token_filters, tokenization

__version__ = "0.0.0"  # Replaced by poetry-dynamic-versioning when deploying

__all__ = [
    "core",
    "filters",
    "Compose",
    "Filter",
    "TokenFilter",
    "Document",
    "Token",
    "deduplication",
    "document_filters",
    "token_filters",
    "tokenization",
]
