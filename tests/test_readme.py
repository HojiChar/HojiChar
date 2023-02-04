from hojichar import Compose, document_filters


def test_rocket_start():
    cleaner = Compose(
        [
            document_filters.JSONLoader(key="text"),
            document_filters.AcceptJapanese(),
            document_filters.DocumentLengthFilter(min_doc_len=0, max_doc_len=1000),
            document_filters.ExampleHojiChar(),
        ]
    )

    assert "こんにちは、<hojichar>" == cleaner('{"text": "こんにちは、"}')
