import numpy as np

from hojichar.core.composition import Compose
from hojichar.core.models import Document
from hojichar.filters.document_filters import DiscardAll, ExampleHojiChar, Identity
from hojichar.filters.tokenization import BlankCharTokenizer


class TestCompose:
    def test_compose(self):
        cleaner = Compose([ExampleHojiChar(), ExampleHojiChar()])
        assert cleaner("") == "<hojichar><hojichar>"

    def test_discard_num_ignore_filtered_option_positive_case(self):
        cleaner = Compose([DiscardAll(), DiscardAll()])
        cleaner("hoge")
        counts = cleaner._statistics.counts
        assert counts["0-DiscardAll"].discard_num == 1
        assert counts["1-DiscardAll"].discard_num == 0

    def test_diff_bytes_ingore_filtered_option_positive_case(self):
        cleaner = Compose(
            [Identity(), DiscardAll(), ExampleHojiChar()],
        )
        cleaner("a")
        counts = cleaner._statistics.counts
        assert counts["0-Identity"].diff_bytes == 0
        assert counts["1-DiscardAll"].diff_bytes == -1
        assert counts["2-ExampleHojiChar"].diff_bytes == 0

    def test_random_apply1(self):
        cleaner = Compose([DiscardAll(p=0.1)], random_state=42)
        count_discard = 0
        for i in range(10000):
            if cleaner.apply(Document("hoge")).is_rejected:
                count_discard += 1
        assert count_discard < 1500

    def test_random_apply2(self):
        rng = np.random.default_rng(42)
        cleaner = Compose([DiscardAll(p=0.1)], random_state=rng)
        count_discard = 0
        for i in range(10000):
            if cleaner.apply(Document("hoge")).is_rejected:
                count_discard += 1
        assert count_discard < 1500

    def test_token_count(self):
        cleaner = Compose([BlankCharTokenizer()])
        cleaner("foo bar")
        assert cleaner.statistics["total_info"]["total_token_num"] == 2
        cleaner("foo hoge")
        assert cleaner.statistics["total_info"]["total_token_num"] == 4
