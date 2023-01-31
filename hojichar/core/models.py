from typing import List


class Token:
    def __init__(self, text: str, is_rejected=False) -> None:
        self.text = text
        self.__original = text
        self.is_rejected = is_rejected

    @property
    def original(self) -> str:
        return self.__original


class Document:
    def __init__(self, text: str, is_rejected=False, tokens=None) -> None:
        self.text = text
        self.__original = text
        self.is_rejected = is_rejected
        self.processed_text = ""
        if tokens is None:
            self.tokens: List[Token] = []

        self.dedup_lsh: List[str] = []

    @property
    def original(self) -> str:
        return self.__original

    def set_tokens(self, tokens: List[str]) -> None:
        self.tokens = [Token(token) for token in tokens]

    def get_tokens(self) -> List[str]:
        return [token.text for token in self.tokens]
