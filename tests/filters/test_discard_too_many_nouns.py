from hojichar import Document
from hojichar.filters.document_filters import DiscardTooManyNouns


def test_discard_too_many_nouns() -> None:
    filter = DiscardTooManyNouns()

    # documents that we want to keep untouched
    assert not filter.apply(Document("自然言語処理大好き！！")).is_rejected
    assert not filter.apply(Document("日本の試合は全部ナイターです")).is_rejected
    assert not filter.apply(Document("NvidiaのGPU大好き。AMDよりも好きかもしれない。")).is_rejected

    # documents to be removed
    garbage_doc_1 = """
    郷土料理. 餃子・点心. 漬物. いろいろな麺類. お好み焼き・焼そば. 味噌汁・スープ. 串カツ・串揚げ. キャンピングカー作り 郷土料理 餃子・点心 居酒屋・ダイニング・お酒処 -
    かおまるっちの大阪食べ歩き日記\n郷土料理. 餃子・点心. 漬物. いろいろな麺類. お好み焼き・焼そば. 味噌汁・スープ. 串カツ・串揚げ. キャンピングカー作り 郷土料理 餃子・点心 居酒屋・ダイニング
    ・お酒処+ラーメン. | ラーメン.+郷土料理. 餃子・点心. 漬物. いろいろな麺類. お好み焼き・焼そば. 味噌汁・スープ. 串カツ・串揚げ.
    """  # noqa

    garbage_doc_2 = """
    ウェブ\n本文: 病院ホームページ検索-秋田県仙北郡/リハビリテーション科\n仙北郡/リハビリテーション科の病院・医院・診療所のサイトを探すなら
    | 仙北郡 | | 秋田市 | | 大館市 | | 男鹿市 | | 潟上市 | | 鹿角市 | | 北秋田市 | | 仙北市 | | 大仙市 | | 能代市 | |
    湯沢市 | | 由利本荘市 | | 横手市 | | 雄勝郡 | | 仙北郡 | | 南秋田郡 | | 山本郡
    """
    assert filter.apply(Document(garbage_doc_1)).is_rejected
    assert filter.apply(Document(garbage_doc_2)).is_rejected
