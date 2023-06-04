import pytest

import hojichar
from hojichar.utils.process import process_iter


class MockExcetion(Exception):
    pass


class MockFilter(hojichar.Filter):
    def apply(self, doc: hojichar.Document) -> hojichar.Document:
        text = doc.text
        if "<reject>" in text:
            doc.is_rejected = True
        if "<value_error>" in text:
            raise ValueError
        if "<mock_error>" in text:
            raise MockExcetion
        doc.text = text
        return doc


@pytest.mark.parametrize(
    "test_data, expected_output",
    [
        (["Line1", "Line2", "Line3"], ["Line1", "Line2", "Line3"]),
        (["Line1", "Line2<reject>", "Line3"], ["Line1", "Line3"]),
        (["Line1", "Line2<value_error>", "Line3"], ["Line1", "Line3"]),
        (["Line1", "Line2<mock_error>", "Line3"], ["Line1", "Line3"]),
    ],
)
def test_process_iter(test_data, expected_output):
    out_iter = process_iter(test_data, MockFilter(), exit_on_error=False)
    assert list(out_iter), expected_output


@pytest.mark.parametrize(
    "test_data, error",
    [
        (["Line1", "Line2<value_error>", "Line3"], ValueError),
        (["Line1", "Line2<mock_error>", "Line3"], MockExcetion),
    ],
)
def test_process_iter_raise(test_data, error):
    out_iter = process_iter(test_data, MockFilter(), exit_on_error=True)
    assert "Line1" == next(out_iter)
    with pytest.raises(error):
        next(out_iter)
