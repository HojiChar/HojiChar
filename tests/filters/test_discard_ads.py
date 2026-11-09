from hojichar import Document
from hojichar.filters.document_filters import BASE_PATH, DiscardAds

# Three keyword matches in 16 characters.
AD_LINE = "求人情報 未経験歓迎 応募資格\n"
# A notice that lists its items with circled numbers, which are in the keyword list.
NOTICE = (
    "令和6年度の事業について、次のとおりお知らせします。\n"
    + "".join(f"{c} 申請書は窓口または郵送で提出してください。\n" for c in "①②③④⑤⑥")
    + "この事業は、地域の子育て世帯を支援するために市が実施するものです。\n" * 40
    + "詳しくは担当課までお問い合わせください。\n" * 3
)


def test_default_decides_by_count() -> None:
    filt = DiscardAds()
    assert not filt.apply(Document(AD_LINE * 4)).is_rejected  # 12 matches
    assert filt.apply(Document(AD_LINE * 5)).is_rejected  # 15 matches
    # A long notice is rejected on the count alone.
    assert len(filt.keyword_pat.findall(NOTICE * 2)) == 18
    assert filt.apply(Document(NOTICE * 2)).is_rejected


def test_min_matches_per_1k_chars() -> None:
    filt = DiscardAds(min_matches_per_1k_chars=10.0)
    # 18 matches in about 3,200 characters (5.6 per 1,000): kept.
    assert not filt.apply(Document(NOTICE * 2)).is_rejected
    # 15 matches in 80 characters: rejected.
    assert filt.apply(Document(AD_LINE * 5)).is_rejected
    # The count condition still applies to short dense documents.
    assert not filt.apply(Document(AD_LINE * 4)).is_rejected


def test_positional_args_unchanged() -> None:
    filt = DiscardAds(BASE_PATH / "dict/advertisement_keywords_ja.txt", 14, 0.5)
    assert filt.max_allow_num == 14
    assert filt.p == 0.5
    assert filt.min_matches_per_1k_chars == 0.0
