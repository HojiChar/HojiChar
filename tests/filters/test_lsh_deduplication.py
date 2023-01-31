import os

from hojichar.core.models import Document
from hojichar.filters.deduplication import GenerateDedupLSH, LSHDeduplicator


class TestLSHDeduplicator:
    temporary_path = "load_blacklist.txt"

    def prepare_blacklist(self):
        deduplicator = LSHDeduplicator(store_blacklist=True)
        d1 = Document("吾輩は猫である。名前はまだ無い。どこで生まれたかとんと見当がつかぬ。")
        d1 = GenerateDedupLSH().apply(d1)
        d2 = Document("吾輩は鳥である。名前はまだ無い。どこで生まれたかとんと見当がつかぬ。")
        d2 = GenerateDedupLSH().apply(d2)

        deduplicator.apply(d1)
        deduplicator.apply(d2)
        with open(self.temporary_path, "w") as fp:
            for lsh in deduplicator.blacklist:
                fp.write(lsh + "\n")

    def test_load_blacklist(self):
        self.prepare_blacklist()

        try:
            deduplicator = LSHDeduplicator(blacklist_path=self.temporary_path)

            d2 = Document("吾輩は鳥である。名前はまだ無い。どこで生まれたかとんと見当がつかぬ。")
            d2 = GenerateDedupLSH().apply(d2)
            d3 = Document("祇園精舎の鐘の声、諸行無常の響きあり。")
            d3 = GenerateDedupLSH().apply(d3)

            assert deduplicator.apply(d2).is_rejected
            assert not deduplicator.apply(d3).is_rejected

        finally:
            os.remove(self.temporary_path)
