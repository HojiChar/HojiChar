import multiprocessing
import pickle
from unittest.mock import Mock

import pytest

import hojichar
from hojichar.core.models import Document
from hojichar.core.parallel import _START_METHOD_ENV_VAR
from hojichar.filters.language_identification import (
    AcceptJapaneseByFastText,
    LanguageIdentificationByFastText,
)


class UnpicklableFastTextModel:
    """Minimal fastText-compatible model representing an unpicklable C++ binding."""

    def __getstate__(self) -> None:
        raise TypeError("cannot pickle fastText model")

    def predict(self, texts: list[str]) -> tuple[list[list[str]], list[list[float]]]:
        labels = [["__label__ja"] if "ほうじ茶" in text else ["__label__en"] for text in texts]
        return labels, [[0.9] for _ in texts]


def test_predict_language_uses_batch_api() -> None:
    filter = LanguageIdentificationByFastText.__new__(LanguageIdentificationByFastText)
    filter.model = Mock()
    filter.model.predict.return_value = ([["__label__ja"]], [[0.9]])

    assert filter._predict_language("ほうじ\n茶") == ("ja", 0.9)
    filter.model.predict.assert_called_once_with(["ほうじ 茶"])


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="Unpicklable filters require the fork start method",
)
def test_unpicklable_fasttext_model_can_run_in_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_START_METHOD_ENV_VAR, raising=False)
    monkeypatch.setattr(multiprocessing, "get_start_method", Mock(return_value=None))
    filter = LanguageIdentificationByFastText.__new__(LanguageIdentificationByFastText)
    hojichar.Filter.__init__(filter)
    filter.language = "ja"
    filter.lang_score_threshold = 0.5
    filter.model = UnpicklableFastTextModel()
    composed_filter = hojichar.Compose([filter])

    with pytest.raises(TypeError, match="cannot pickle fastText model"):
        pickle.dumps(composed_filter)

    documents = [hojichar.Document("ほうじ茶"), hojichar.Document("English text")]
    with hojichar.Parallel(composed_filter, num_jobs=2, ordered=True) as pfilter:
        processed_documents = list(pfilter.imap_apply(iter(documents)))

    assert [document.is_rejected for document in processed_documents] == [False, True]


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
