import numpy as np

from hojichar.core.composition import Compose
from hojichar.core.models import Document, Statistics
from hojichar.filters.document_filters import DiscardAll, ExampleHojiChar, Identity


class TestCompose:
    def test_compose(self) -> None:
        cleaner = Compose([ExampleHojiChar(), ExampleHojiChar()])
        assert cleaner("") == "<hojichar><hojichar>"

    def test_discard_num_ignore_filtered_option_positive_case(self) -> None:
        cleaner = Compose([DiscardAll(), DiscardAll()])
        cleaner("hoge")
        stats = cleaner.get_total_statistics()
        assert Statistics.get_filter("0-DiscardAll", stats).discard_num == 1
        assert Statistics.get_filter("1-DiscardAll", stats).discard_num == 0

    def test_diff_bytes_ingore_filtered_option_positive_case(self) -> None:
        cleaner = Compose(
            [Identity(), DiscardAll(), ExampleHojiChar()],
        )
        cleaner("a")
        stats = cleaner.get_total_statistics()
        assert Statistics.get_filter("0-Identity", stats).diff_bytes == 0
        assert Statistics.get_filter("1-DiscardAll", stats).diff_bytes == -1
        assert Statistics.get_filter("2-ExampleHojiChar", stats).diff_bytes == 0

    def test_random_apply1(self) -> None:
        cleaner = Compose([DiscardAll(p=0.1)], random_state=42)
        count_discard = 0
        for i in range(10000):
            if cleaner.apply(Document("hoge")).is_rejected:
                count_discard += 1
        assert count_discard < 1500

    def test_random_apply2(self) -> None:
        rng = np.random.default_rng(42)
        cleaner = Compose([DiscardAll(p=0.1)], random_state=rng)
        count_discard = 0
        for i in range(10000):
            if cleaner.apply(Document("hoge")).is_rejected:
                count_discard += 1
        assert count_discard < 1500

    def test_char_count(self) -> None:
        cleaner = Compose([Identity()])
        cleaner("foo bar")
        stats = cleaner.get_total_statistics()
        assert Statistics.get_filter("Total", stats).input_chars == 7

    def test_expand_filters(self) -> None:
        cleaner1 = Compose([ExampleHojiChar(), ExampleHojiChar()])
        cleaner2 = Compose([ExampleHojiChar(), cleaner1])
        cleaner2("hello")
        stats = cleaner2.get_total_statistics()
        assert len(stats) == 4

    def test_apply_batch_returns_processed_documents(self) -> None:
        docs = [Document(""), Document("foo")]
        cleaner = Compose([ExampleHojiChar()])
        processed = cleaner.apply_batch(docs)
        # Document オブジェクトの数, テキスト変換の確認
        assert len(processed) == 2
        assert processed[0].text == "<hojichar>"
        assert processed[1].text == "foo<hojichar>"

    def test_apply_stream_yields_and_cleans_extras(self) -> None:
        docs = [Document("bar"), Document("baz")]
        cleaner = Compose([ExampleHojiChar()])
        out = list(cleaner.apply_stream(iter(docs)))
        # テキストの加工確認
        assert [d.text for d in out] == ["bar<hojichar>", "baz<hojichar>"]
        # extras に一時キーが残っていないこと
        for d in out:
            for key in ("__start_ns", "__input_bytes", "__input_chars", "__orig_rejected"):
                assert key not in d.extras

    def test_apply_and_stream_with_discard_all_and_batch_equivalence(self) -> None:
        # 同じフィルタ列で apply_batch と apply_stream が同じ結果を返すこと
        docs = [Document("x"), Document("y"), Document("z")]
        cleaner = Compose([ExampleHojiChar(), DiscardAll()])
        batch_out = cleaner.apply_batch(docs)
        stream_out = list(cleaner.apply_stream(iter(docs)))
        assert [d.text for d in batch_out] == [d.text for d in stream_out]
        # 最後の DiscardAll で拒否されるためすべて空文字
        assert all(d.is_rejected or d.text == "" for d in stream_out)

    def test_random_state_propagation_to_nested_filters(self) -> None:
        # nested Compose の RNG が同一シードを共有しているか
        rng = np.random.default_rng(123)
        inner = Compose([DiscardAll(p=0.5)])
        outer = Compose([inner], random_state=rng)
        # outer の RNG を変えても inner に伝播される
        # 100 回試行で内側フィルタの拒否率がおおむね p=0.5 に近いことを確認
        rejections = sum(1 for _ in range(1000) if outer.apply(Document("hoge")).is_rejected)
        assert 300 < rejections < 700  # 50% ±20%
