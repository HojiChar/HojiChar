from hojichar.core.models import Document
from hojichar.filters import tokenization


class TestMergeTokens:
    def test_merge(self):
        doc = Document("おはよう。おやすみ。ありがとう。さよなら。")
        tokenizer = tokenization.SentenceTokenizer()
        transformed_doc = tokenizer.apply(doc)
        assert [
            "おはよう。",
            "おやすみ。",
            "ありがとう。",
            "さよなら。",
        ] == transformed_doc.get_tokens()
        assert "おはよう。おやすみ。ありがとう。さよなら。" == tokenization.MergeTokens().apply(transformed_doc).text
        assert "おはよう。\nおやすみ。\nありがとう。\nさよなら。" == tokenization.MergeTokens("\n").apply(transformed_doc).text
