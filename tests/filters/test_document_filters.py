import json

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

    def test_apply_loads_embedded_extras(self):
        data = '{"text": "hello", "extras": {"key1": "val1", "nested": {"foo": "bar"}}}'
        doc = Document(data)
        loaded = document_filters.JSONLoader().apply(doc)
        assert loaded.text == "hello"
        assert loaded.extras["key1"] == "val1"
        assert loaded.extras["nested"] == {"foo": "bar"}

    def test_apply_extra_keys_merge_with_existing_extras(self):
        data = '{"text": "hello", "url": "https://example.com"}'
        doc = Document(data, extras={"existing": "value"})
        loaded = document_filters.JSONLoader(extra_keys=["url"]).apply(doc)
        assert loaded.text == "hello"
        assert loaded.extras == {"existing": "value", "url": "https://example.com"}


class TestJSONDumper:
    def test_export_extras_roundtrip(self):
        loader_input = '{"text": "hello", "extras": {"key1": "val1"}}'
        doc = document_filters.JSONLoader().apply(Document(loader_input))
        doc.extras["added_in_pipeline"] = 123

        dumped = document_filters.JSONDumper(export_extras=True).apply(doc)
        payload = json.loads(dumped.text)

        assert payload["text"] == "hello"
        assert payload["extras"]["key1"] == "val1"
        assert payload["extras"]["added_in_pipeline"] == 123
