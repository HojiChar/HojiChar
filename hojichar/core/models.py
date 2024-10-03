from typing import Any, Dict, List, Optional


class Token:
    def __init__(self, text: str, is_rejected: bool = False) -> None:
        self.text = text
        self.__original = text
        self.is_rejected = is_rejected

    @property
    def original(self) -> str:
        return self.__original

    def __str__(self) -> str:
        return self.text


class Document:
    def __init__(
        self,
        text: str,
        is_rejected: bool = False,
        tokens: Optional[List[Token]] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.text = text
        self.__original = text
        self.is_rejected = is_rejected
        if tokens is None:
            self.tokens: List[Token] = []

        if extras is None:
            self.extras: Dict[str, Any] = {}
        else:
            self.extras = extras

        self.dedup_lsh: List[str] = []
        self.reject_reason: Dict[str, Any] = {}

    @property
    def original(self) -> str:
        return self.__original

    def set_tokens(self, tokens: List[str]) -> None:
        self.tokens = [Token(token) for token in tokens]

    def get_tokens(self) -> List[str]:
        return [token.text for token in self.tokens]

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"Document(text={self.text!r}, is_rejected={self.is_rejected}, extras={self.extras})"  # noqa
