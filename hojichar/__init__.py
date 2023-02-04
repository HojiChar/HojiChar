"""
.. include:: ../README.md
"""
from .core.composition import Compose
from .core.filter_interface import Filter, TokenFilter
from .core.models import Document, Token
from .filters import deduplication, document_filters, token_filters, tokenization

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
