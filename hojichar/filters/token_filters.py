import re
from typing import Any

from hojichar.core.filter_interface import TokenFilter
from hojichar.core.models import Token


class TokenAddDebagTag(TokenFilter):
    """トークン末尾にデバッグ用のタグを追加します."""

    def apply(self, token: Token) -> Token:
        """
        >>> TokenAddDebagTag()("hello")
        'hello<sep>'
        """
        token.text += "<sep>"
        return token


class SEOTokenRemover(TokenFilter):
    """
    The process is migrated from legacy code.
    I couldn't understand what this process was about, mainly because
    the regex pattern is too complex.
    """

    def __init__(self, min_average_seo_char_length: int = 5, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.token_split_pat = re.compile(r"\ |-|・|,")
        self.min_average_seo_char_length = min_average_seo_char_length
        self.replace_pat = re.compile(
            r"\-{5,},@[a-zA-Z0-9]+,[#\$\%\-]{4,},[＿=#\$\%\-]{4,}[\ ]*.+?[\ ]*[＿=#\$\%\-]{4,}|★[…━]+★"  # noqa
        )

    def apply(self, token: Token) -> Token:
        seo_words = self.token_split_pat.split(token.text.strip())
        n_words = len(seo_words)
        if n_words == 0:
            return token
        avg_char_len = len(token.text) / n_words

        if avg_char_len <= self.min_average_seo_char_length:
            return token

        replace_patterns = self.replace_pat.search(token.text)
        if replace_patterns is not None:
            token.text = token.text.replace(replace_patterns.group(0), "", 1)

        return token
