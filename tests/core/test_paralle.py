from __future__ import annotations

import json

import pytest

import hojichar
from hojichar.core.parallel import Parallel
from hojichar.filters.document_filters import JSONDumper, JSONLoader


class RaiseKeywords(hojichar.Filter):
    def apply(self, document: hojichar.Document) -> hojichar.Document:
        text = document.text
        if "<raise>" in text:
            raise
        return document


@pytest.mark.parametrize("num_jobs", [1, 4, None])
def test_processed_docs_count(num_jobs: int | None) -> None:
    documents = [hojichar.Document(json.dumps({"text": f"doc_{i}"})) for i in range(10)]
    filter = hojichar.Compose([JSONLoader(), JSONDumper()])

    with Parallel(filter, num_jobs=num_jobs) as pfilter:
        list(pfilter.imap_apply(iter(documents)))
        assert pfilter.statistics_obj.total_info.processed_num == 10


@pytest.mark.parametrize("num_jobs", [1, 4, None])
def test_processed_docs_equality(num_jobs: int | None) -> None:
    documents = [hojichar.Document(json.dumps({"text": f"doc_{i}"})) for i in range(10)]
    filter = hojichar.Compose([JSONLoader(), JSONDumper()])

    with Parallel(filter, num_jobs=num_jobs) as pfilter:
        processed_docs = list(pfilter.imap_apply(iter(documents)))
        assert set(str(s) for s in processed_docs) == set(str(s) for s in documents)


@pytest.mark.parametrize("num_jobs", [1, 4, None])
def test_filter_statistics_increment(num_jobs: int | None) -> None:
    documents = [hojichar.Document(json.dumps({"text": f"doc_{i}"})) for i in range(10)]
    filter = hojichar.Compose([JSONLoader(), JSONDumper()])

    with Parallel(filter, num_jobs=num_jobs) as pfilter:
        list(pfilter.imap_apply(iter(documents)))

    with Parallel(filter, num_jobs=num_jobs) as pfilter:
        list(pfilter.imap_apply(iter(documents)))

    assert filter.statistics_obj.total_info.processed_num == 20


@pytest.mark.parametrize("num_jobs", [1, 4, None])
def test_parallel_with_error_handling(num_jobs: int | None) -> None:
    documents = [hojichar.Document(f"<raise>_{i}") for i in range(10)]
    error_filter = hojichar.Compose([RaiseKeywords()])

    with pytest.raises(Exception):
        with Parallel(error_filter, num_jobs=num_jobs) as pfilter:
            list(pfilter.imap_apply(iter(documents)))
            pfilter.statistics_obj.total_info.processed_num == 0
    assert error_filter.statistics_obj.total_info.processed_num == 0

    with Parallel(error_filter, num_jobs=2, ignore_errors=True) as pfilter:
        processed_docs = list(pfilter.imap_apply(iter(documents)))
        assert list(str(s) for s in processed_docs) == [""] * 10
        pfilter.statistics_obj.total_info.processed_num == 0
    assert error_filter.statistics_obj.total_info.processed_num == 0
