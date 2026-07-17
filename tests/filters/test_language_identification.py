from unittest.mock import Mock

import pytest

from hojichar.core.models import Document
from hojichar.filters.language_identification import (
    AcceptJapaneseByFastText,
    LanguageIdentificationByFastText,
)


def test_predict_language_uses_batch_api() -> None:
    filter = LanguageIdentificationByFastText.__new__(LanguageIdentificationByFastText)
    filter.model = Mock()
    filter.model.predict.return_value = ([["__label__ja"]], [[0.9]])

    assert filter._predict_language("ほうじ\n茶") == ("ja", 0.9)
    filter.model.predict.assert_called_once_with(["ほうじ 茶"])


@pytest.mark.download_test
def test_accept_japanese_by_fasttext() -> None:
    filter = AcceptJapaneseByFastText()

    # Japanese text
    assert not filter.apply(Document("ほうじ茶")).is_rejected
    assert not filter.apply(Document("自然言語処理さいこう！")).is_rejected
    assert not filter.apply(Document("NvidiaのGPU大好き。AMDよりも好きかもしれない。")).is_rejected

    # Non-japanese text
    assert filter.apply(Document("I am an NLPer")).is_rejected
    assert filter.apply(Document("快三手机投注平台代理")).is_rejected
    assert filter.apply(Document("Carrément dernier vin meilleur mais boulangerie.")).is_rejected
