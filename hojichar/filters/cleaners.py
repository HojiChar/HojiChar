import logging
import pathlib
import re

import hojichar
from hojichar.core.filter_interface import Filter
from hojichar.core.models import Document, Token

BASE_PATH = pathlib.Path(hojichar.__path__[0])
logger = logging.getLogger(__name__)


class ApplyDiscard(Filter):
    """
    上流フィルタで破棄された`Document`を空文字列にします.

    `Document.is_rejected=True` の ドキュメントは無視されるため,
    このフィルタを `Compose` のコンストラクタに渡しても動作しません.
    このフィルタは主に`Compose` 内部や, `discard_filtered=False` を指定
    したデバッグ時などに利用されます.
    """

    def apply(self, document: Document) -> Document:
        """
        >>> ApplyDiscard().apply(Document(text="hello", is_rejected=True)).text
        ''
        """
        if document.is_rejected:
            document.text = ""

        return document


class DocumentLengthFilter(Filter):
    """
    `min_doc_len`, `max_doc_len` で指定した上限・下限の範囲内にないドキュメントを破棄します.
    デフォルトでは 200字 以上 50000字以内のテキストが受理されます.
    """

    def __init__(self, min_doc_len=None, max_doc_len=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_doc_len = min_doc_len
        self.max_doc_len = max_doc_len

    def apply(self, doc: Document) -> Document:
        """
        >>> DocumentLengthFilter(min_doc_len=5).apply(Document("1234")).is_rejected
        True
        """
        doc_len = len(doc.text)
        if self.min_doc_len is not None:
            if doc_len < self.min_doc_len:
                doc.is_rejected = True
        if self.max_doc_len is not None:
            if self.max_doc_len < doc_len:
                doc.is_rejected = True
        return doc


class NgWordsFilterJa(Filter):
    """
    日本語のNGワード(および不適切語)を含む文書を破棄します.
    `dict_path` で指定したファイルから, キーワードのリストを得ます.
    ファイルは単語が改行で羅列されたテキストファイルです.

    `ignore_confused` を `True` にすると,
    偽陽性を軽減するために, カタカナのNGワードは前後にカタカナが無い場合のみNG判定されます.
    デフォルト値は `False` です.
    """

    def __init__(self, dict_path, ignore_confused=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(dict_path, encoding="utf-8") as fp:
            ng_words = fp.read().split("\n")
        ng_words = [w.strip() for w in ng_words if not len(w) == 0]

        if ignore_confused:
            words_katakana = []
            words_not_katakana = []
            for w in ng_words:
                if re.fullmatch(r"[ァ-ヴー]+", w):
                    words_katakana.append(re.escape(w))
                else:
                    words_not_katakana.append(re.escape(w))
            katakana_pat = "|".join(words_katakana)
            katakana_pat = rf"(?<![ァ-ヴー])({katakana_pat})(?![ァ-ヴー])"
            pat = "|".join(words_not_katakana) + "|" + katakana_pat
            self.keyword_pat = re.compile(pat)
        else:
            ng_words = [re.escape(w) for w in ng_words]
            pat = "|".join(ng_words)
            self.keyword_pat = re.compile(pat)

    def apply(self, doc: Document) -> Document:
        if self.keyword_pat.search(doc.text):
            doc.is_rejected = True
        return doc


class NgWordsFilterEn(Filter):
    """
    英語のNGワード(および不適切語)を含む文書を破棄します.
    `dict_path` で指定したファイルから, キーワードのリストを得ます.
    ファイルは単語が改行で羅列されたテキストファイルです.
    """

    def __init__(self, dict_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(dict_path, encoding="utf-8") as fp:
            ng_words = fp.read().split("\n")
        ng_words = [re.escape(w.strip()) for w in ng_words if not len(w) == 0]
        pat = "|".join(ng_words)
        # 英語のパターンにマッチするようにしている, \s[単語]\s や [単語]. [単語], などにマッチ.
        self.keyword_pat = re.compile(rf"(?:^| )({pat})(?:( |,|\.)|$)", re.IGNORECASE)

    def apply(self, doc: Document) -> Document:
        if self.keyword_pat.search(doc.text):
            doc.is_rejected = True
        return doc


class DiscardAdultContentJa(NgWordsFilterJa):
    """
    日本語のアダルトキーワード(および不適切語)を含む文書を破棄します.
    `dict_path` で指定したファイルから, キーワードのリストを得ます.
    ファイルは単語が改行で羅列されたテキストファイルです.
    デフォルトの`dict_path` は /hojichar/dict/adult_keywords_ja.txt です.
    """

    def __init__(self, dict_path=BASE_PATH / "dict/adult_keywords_ja.txt", *args, **kwargs):
        super().__init__(dict_path, *args, **kwargs)

    def apply(self, doc: Document) -> Document:
        """
        >>> DiscardAdultContentJa().apply(Document("<TEST_STRING_OF_ADULT_KEYWORD>")).is_rejected
        True

        >>> DiscardAdultContentJa().apply(Document("ほうじ茶")).is_rejected
        False

        挙動は正しいが誤検知しているケース. 他にも, サック in リュックサック,
        >>> DiscardAdultContentJa().apply(Document("アスパラガス"))\
            .is_rejected # Matching with NG keyword "アス"
        True
        """
        return super().apply(doc)


class DiscardAdultContentEn(NgWordsFilterEn):
    """
    英語のアダルトキーワード(および不適切語)を含む文書を破棄します.
    `dict_path` で指定したファイルから, キーワードのリストを得ます.
    ファイルは単語が改行で羅列されたテキストファイルです.
    デフォルトの`dict_path` は /hojichar/dict/adult_keywords_en.txt です.
    """

    def __init__(self, dict_path=BASE_PATH / "dict/adult_keywords_en.txt", *args, **kwargs):
        super().__init__(dict_path, *args, **kwargs)

    def apply(self, doc: Document) -> Document:
        """
        >>> DiscardAdultContentEn().apply(Document("<TEST_STRING_OF_ADULT_KEYWORD>")).is_rejected
        True

        >>> DiscardAdultContentEn().apply(Document("hojichar")).is_rejected
        False
        """
        return super().apply(doc)


class DiscardDiscriminationContentJa(NgWordsFilterJa):
    """
    日本語の差別キーワード(および不適切語)を含む文書を破棄します.
    `dict_path` で指定したファイルから, キーワードのリストを得ます.
    ファイルは単語が改行で羅列されたテキストファイルです.
    デフォルトの`dict_path` は /hojichar/dict/discrimination_keywords_ja.txt です.
    """

    def __init__(
        self,
        dict_path=BASE_PATH / "dict/discrimination_keywords_ja.txt",
        *args,
        **kwargs,
    ):
        super().__init__(dict_path, *args, **kwargs)

    def apply(self, doc: Document) -> Document:
        """
        >>> DiscardDiscriminationContentJa().\
            apply(Document("<TEST_STRING_OF_DISCRIMINATION_KEYWORD>")).is_rejected
        True

        >>> DiscardDiscriminationContentJa().apply(Document("ほうじ茶")).is_rejected
        False
        """
        return super().apply(doc)


class DiscardViolenceContentJa(NgWordsFilterJa):
    """
    日本語の暴力・脅迫を示唆するキーワードを含む文書を破棄します.
    `dict_path` で指定したファイルから, キーワードのリストを得ます.
    ファイルは単語が改行で羅列されたテキストファイルです.
    デフォルトの`dict_path` は /hojichar/dict/violence_keywords_ja.txt です.
    """

    def __init__(self, dict_path=BASE_PATH / "dict/violence_keywords_ja.txt", *args, **kwargs):
        super().__init__(dict_path, *args, **kwargs)

    def apply(self, doc: Document) -> Document:
        """
        >>> DiscardViolenceContentJa().\
            apply(Document("<TEST_STRING_OF_VIOLENCE_KEYWORD>")).is_rejected
        True

        >>> DiscardViolenceContentJa().apply(Document("ほうじ茶")).is_rejected
        False
        """
        return super().apply(doc)


class DiscardBBSComments(Filter):
    """
    正規表現 "BBS Patern" に `max_allow_num` 回よりたくさんマッチする文書を破棄します.
    `max_allow_num` のデフォルト値は14です.
    正規表現 "BBS Patern" は下記のリンクで検証可能です.
    https://regex101.com/r/ybQvL2/1
    """

    def __init__(self, max_allowed_num=14, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_allowed_num = max_allowed_num
        self.keyword_pat = re.compile(
            r"\d{4}[年\.\-\/][\ ]*\d{1,2}[月\.\-\/][\ ]*\d{1,2}[日]*|コメント|SOLD OUT|レビュー|投稿|ページ|\([月火水木金土日]\)|質問|\d+話|楽天市場|-"  # noqa
        )

    def apply(self, doc):
        """
        >>> DiscardBBSComments().apply(Document("楽天市場 質問 投稿 コメント レビュー "*3)).is_rejected
        True

        >>> DiscardBBSComments().apply(Document("鏡餅")).is_rejected
        False
        """
        bbs_factor = self.keyword_pat.findall(doc.text)
        if len(bbs_factor) > self.max_allowed_num:
            doc.is_rejected = True
        return doc


class DiscardAds(Filter):
    """
    主に広告キーワーを`max_allow_num`より多く含む文書を破棄します.
    デフォルトで`max_allow_num` は14です.
    `dict_path` で指定したファイルから, 広告キーワードのリストを得ます.
    ファイルは単語が改行で羅列されたテキストファイルです.
    デフォルトの`dict_path` は /hojichar/dict/advertisement_keywords_ja.txt です.
    """

    def __init__(
        self,
        dict_path=BASE_PATH / "dict/advertisement_keywords_ja.txt",
        max_allowed_num=14,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_allow_num = max_allowed_num
        with open(dict_path, encoding="utf-8") as fp:
            ng_words = fp.read().split("\n")
        ng_words = [re.escape(w.strip()) for w in ng_words if not len(w) == 0]
        pat = r"|".join(ng_words)
        self.keyword_pat = re.compile(pat)

    def apply(self, doc: Document) -> Document:
        """
        >>> DiscardAds().apply(Document("お問い合わせください 営業時間 よくある質問"*5)).is_rejected
        True

        >>> DiscardAds().apply(Document("おはよう")).is_rejected
        False
        """
        ads_factor = self.keyword_pat.findall(doc.text)
        if len(ads_factor) > self.max_allow_num:
            doc.is_rejected = True
        return doc


class AcceptJapanese(Filter):
    """
    日本語でないドキュメントを破棄します. 日本語判定は次の手順で行われます.
        1. テキストを左から`lookup_size` (デフォルトで50字) 参照し,
        ひらがな・カタカナが存在すれば日本語と判定する.
    """

    def __init__(self, lookup_size=50, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lookup_size = 50
        self.hiragana_katakana_pat = re.compile(r"[ぁ-んァ-ン]")

    def apply(self, doc: Document) -> Document:
        """
        >>> AcceptJapanese().apply(Document("This is English document")).is_rejected
        True

        >>> AcceptJapanese().apply(Document("a"*50 + "あ")).is_rejected
        True

        >>> AcceptJapanese().apply(Document("ほうじ茶")).is_rejected
        False
        """
        if not self.hiragana_katakana_pat.search(doc.text[: self.lookup_size]):
            doc.is_rejected = True
        return doc


class DiscardRareKuten(Filter):
    """
    日本語でないドキュメントを破棄します. 日本語判定は次の手順で行われます
    ドキュメントを句点"。"で区切り, 平均文長が
    `max_avarage_sentence_length` より長い場合は破棄します.
    `max_avarage_sentence_length` のデフォルト値は100です.
    このフィルタは, 文章中の句点の割合が少なすぎるドキュメントを破棄します.
    """

    def __init__(self, max_average_sentence_length=100, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_average_sentence_length = max_average_sentence_length
        self.kuten_pat = re.compile(r"。")

    def apply(self, doc: Document) -> Document:
        """
        >>> DiscardRareKuten(max_average_sentence_length=4).apply(Document("おはよ。")).is_rejected
        False
        >>> DiscardRareKuten(max_average_sentence_length=4).apply(Document("おはよう。")).is_rejected
        True
        """
        kuten_lst = self.kuten_pat.findall(doc.text)
        min_kuten_num = len(doc.text) / self.max_average_sentence_length
        if len(kuten_lst) < min_kuten_num:
            doc.is_rejected = True
        return doc


class HeaderFooterTagsRemover(Filter):
    """
    ドキュメントの冒頭・末尾のトークンを調査し, ヘッダー・フッダー的な
    タグが存在していた場合, そのトークンを除去します.

    このフィルタを通す前に, 事前にセンテンスレベルにトーカナイズしておいてください.
    このフィルタでは Document.token にのみ変更が加えられるので, 出力前 あるいは 下流フィルタで
    Document.text に変更を加える前にトークンをマージしておいてください.
    """

    def __init__(
        self,
        dict_path=BASE_PATH / "dict/header_footer_keywords_ja.txt",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        with open(dict_path) as fp:
            keywords = fp.read().split("\n")
        keywords = [re.escape(w.strip()) for w in keywords if not len(w) == 0]
        self.keyword_pat = re.compile(r"|".join(keywords))

    def apply(self, doc: Document) -> Document:
        if len(doc.tokens) == 0:
            return doc

        lookup_size = 0
        if 1 <= len(doc.tokens) < 4:
            lookup_size = 1
        elif 4 <= len(doc.tokens) < 6:
            lookup_size = 2
        elif 6 <= len(doc.tokens):
            lookup_size = 3

        for i in range(lookup_size):
            if self.should_drop_token(doc.tokens[i]):
                doc.tokens[i].is_rejected = True
            if self.should_drop_token(doc.tokens[-(i + 1)]):
                doc.tokens[i].is_rejected = True

        return doc

    def should_drop_token(self, token: Token) -> bool:
        """
        >>> HeaderFooterTagsRemover().should_drop_token(Token("<TEST_STRING_OF_KEYWORD>"))
        True

        >>> HeaderFooterTagsRemover().should_drop_token(Token("ほうじ茶"))
        False

        Comment.
        Legacy code removed a pattern r"« _ | Main | _ »" .
        In the pattern, "|" is not escaped, so **ANY** string was eliminated.
        It seems unintended behavior, so I fix this.
        """
        if self.keyword_pat.match(token.text):
            return True
        else:
            return False


class MaskPersonalInformation(Filter):
    """
    ドキュメントに含まれる電話番号・電子メールアドレスを一部マスキングします.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.phone_pat = re.compile(
            r"((0|\+\d{1,3}[- ]?)(\d{2}[- ]?\d{4}[- ]?|\d[- ]?\d{4}[- ]?|\d{2}[- ]?\d{3}[- ]?|\d{3}[- ]?\d{2}[- ]?|\d{4}[- ]?\d{1}[- ]?))\d{4}"  # noqa
        )
        self.email_pat = re.compile(
            r"[a-zA-Z0-9!#$%&'*+\-/=?^_`{|}~.]+@[A-Za-z0-9!#$%&'*+\-/=?^_`{|}~.]+(\.[A-Za-z0-9\-]+)"  # noqa
        )

    def apply(self, doc: Document) -> Document:
        """
        >>> MaskPersonalInformation()('06-1234-5678')
        '06-1234-XXXX'
        >>> MaskPersonalInformation()('075-123-4567')
        '075-123-XXXX'
        >>> MaskPersonalInformation()('0166-12-3456')
        '0166-12-XXXX'
        >>> MaskPersonalInformation()('09808-1-2345')
        '09808-1-XXXX'
        >>> MaskPersonalInformation()('090-1234-5678')
        '090-1234-XXXX'
        >>> MaskPersonalInformation()('0751234567')
        '075123XXXX'
        >>> MaskPersonalInformation()('08012345678')
        '0801234XXXX'
        >>> MaskPersonalInformation()('連絡は075-123-4567 まで')
        '連絡は075-123-XXXX まで'
        >>> MaskPersonalInformation()('+81-80-1234-5678')
        '+81-80-1234-XXXX'
        >>> MaskPersonalInformation()('+818012345678')
        '+81801234XXXX'
        >>> MaskPersonalInformation()('hogehoge@example.com')
        'xxxx@yyy.com'
        >>> MaskPersonalInformation()('何かあれば hogehoge@example.ne.jp まで連絡')
        '何かあれば xxxx@yyy.jp まで連絡'
        """
        text = self.phone_pat.sub(r"\1XXXX", doc.text)
        text = self.email_pat.sub(r"xxxx@yyy\1", text)
        doc.text = text
        return doc


if __name__ == "__main__":
    import doctest

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s]%(name)s:%(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.DEBUG)

    doctest.testmod()
