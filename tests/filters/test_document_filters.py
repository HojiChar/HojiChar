from hojichar.core.models import Document
from hojichar.filters import tokenization


class TestSentenceTokenizer:
    def test_tokenize(self):
        doc = Document("おはよう。おやすみ。ありがとう。さよなら。")
        tokenizer = tokenization.SentenceTokenizer()
        transfomed_doc = tokenizer.apply(doc)
        assert [
            "おはよう。",
            "おやすみ。",
            "ありがとう。",
            "さよなら。",
        ] == transfomed_doc.get_tokens()

        doc = Document("おはよう。おやすみ。ありがとう。さよなら")
        tokenizer = tokenization.SentenceTokenizer()
        transfomed_doc = tokenizer.apply(doc)
        assert [
            "おはよう。",
            "おやすみ。",
            "ありがとう。",
            "さよなら",
        ] == transfomed_doc.get_tokens()
