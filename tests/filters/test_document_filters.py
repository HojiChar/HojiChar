from hojichar.core.models import Document
from hojichar.filters import document_filters, tokenization


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


class TestJSONLoader:
    def test_apply(self):
        data = '{"text": "おはよう。おやすみ。ありがとう。さよなら。", "url": "https://example.com", "title": "example"}'
        doc = Document(data)
        loaded = document_filters.JSONLoader().apply(doc)
        assert loaded.text == "おはよう。おやすみ。ありがとう。さよなら。"
        assert loaded.extras == {}

        doc = Document(data)
        loaded = document_filters.JSONLoader(extra_keys=["url", "title"]).apply(doc)
        assert loaded.text == "おはよう。おやすみ。ありがとう。さよなら。"
        assert loaded.extras["url"] == "https://example.com"
        assert loaded.extras["title"] == "example"
