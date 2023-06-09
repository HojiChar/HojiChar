from typing import List

from hojichar.core.filter_interface import Filter
from hojichar.core.models import Document


class BlankCharTokenizer(Filter):
    """
    Tokenizer の実装例です.
    ドキュメントを空白文字でトークンに分割します.
    """

    def apply(self, document: Document) -> Document:
        tokens = self.tokenize(document.text)
        document.set_tokens(tokens)
        return document

    def tokenize(self, text: str) -> List[str]:
        """
        >>> BlankCharTokenizer().tokenize("hello world")
        ['hello', 'world']
        """
        return text.split()


class MergeTokens(Filter):
    """
    Merger の実装例です.
    破棄されていないトークンを結合し, Document を更新します.
    """

    def merge(self, tokens: List[str]) -> str:
        """
        >>> MergeTokens().merge(["hoo", "bar"])
        'hoobar'
        """
        return "".join(tokens)

    def apply(self, document: Document) -> Document:
        remained_tokens = [token.text for token in document.tokens if not token.is_rejected]
        document.text = self.merge(remained_tokens)
        return document


class SentenceTokenizer(Filter):
    """
    日本語を想定したセンテンス単位のトーカナイザです.
    句点`。`で文章を区切ります. これだけでは実際の日本語テキストで不十分な例が多くある
    (句点にピリオドが用いられる, 会話のカギカッコ内で句点が用いられるなど)ため,
    将来的には適切なセンテンス単位のトーカナイザに置き換えられるべきです.
    """

    def apply(self, document: Document) -> Document:
        tokens = self.tokenize(document.text)
        document.set_tokens(tokens)
        return document

    def tokenize(self, text: str) -> List[str]:
        """
        >>> SentenceTokenizer().tokenize("おはよう。おやすみ。ありがとう。さよなら。")
        ['おはよう。', 'おやすみ。', 'ありがとう。', 'さよなら。']

        >>> SentenceTokenizer().tokenize("さよなら。また来週")
        ['さよなら。', 'また来週']
        """
        tokens = text.split("。")
        if len(tokens) > 1:
            if text.endswith("。"):
                tokens = [token + "。" for token in tokens[:-1]]
            else:
                last = tokens[-1]
                tokens = [token + "。" for token in tokens]
                tokens[-1] = last

        return tokens
