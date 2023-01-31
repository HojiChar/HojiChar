# flake8: noqa
"""
ドキュメント単位のフィルタの実装です.
このモジュールは廃止され, 機能別に cleaneras, normalization, tokenization などに移行しています.
"""
import copy
import json
import logging
import pathlib
import re
import unicodedata
import warnings
from typing import Callable, List

import mmh3  # type: ignore

import hojichar
from hojichar.core.filter_interface import Filter
from hojichar.core.models import Document, Token

BASE_PATH = pathlib.Path(hojichar.__path__[0])
logger = logging.getLogger(__name__)


class ExampleHojiChar(Filter):
    """基本的なフィルタの実装例です. 末尾に'<hojichar>'を追加します."""

    def apply(self, document: Document) -> Document:
        """
        >>> ExampleHojiChar()("hello, world")
        'hello, world<hojichar>'
        """
        document.text += "<hojichar>"
        return document


class ExampleDiscardDocumentContainKeyword(Filter):
    """特定のキーワードを持つドキュメントを破棄するようなフィルタの実装例です."""

    def __init__(self, keyword: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keyword = keyword

    def apply(self, document: Document) -> Document:
        """
        >>> ExampleDiscardDocumentContainKeyword("バカ").apply(Document("あいつはバカだ")).is_rejected
        True
        """
        if self.keyword in document.text:
            document.is_rejected = True
        return document


class BlankCharTokenizer(Filter):
    """
    このクラスは hojihca.filters.tokenization.BlankCharTokenizer に移動しました.

    Tokenizer の実装例です.
    ドキュメントを空白文字でトークンに分割します.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            "hojichar.filters.document_filter.BlankCharTokenizer は廃止されます. \
        hojihca.filters.tokenization.BlankCharTokenizer に移行しました",
            FutureWarning,
        )

    def apply(self, document: Document) -> Document:
        tokens = self.tokenize(document.text)
        document.set_tokens(tokens)
        return document

    def tokenize(self, text) -> List[str]:
        """
        >>> BlankCharTokenizer().tokenize("hello world")
        ['hello', 'world']
        """
        return text.split()


class MergeTokens(Filter):
    """
    このクラスは hojihca.filters.tokenization.MergeTokens に移動しました.

    Merger の実装例です.
    破棄されていないトークンを結合し, Document を更新します.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            "hojichar.filters.document_filter.MergeTokens は廃止されます. \
        hojihca.filters.tokenization.MergeTokens に移行しました",
            FutureWarning,
        )

    def merge(self, tokens: List[str]) -> str:
        """
        >>> MergeTokens().merge(["hoo", "bar"])
        'hoobar'
        """
        return "".join(tokens)

    def apply(self, document: Document) -> Document:
        remained_tokens = [token.text for token in document.tokens if not token.is_rejected]
        document.text = self.merge(remained_tokens)
        return document


class Identity(Filter):
    """何も変化を加えないフィルタです. テスト・デバッグに用いられます."""

    def apply(self, document):
        return document


class DiscardAll(Filter):
    """
    すべてのドキュメントを破棄するフィルタです.
    テスト・デバッグに用いられます.
    """

    def apply(self, document):
        document.is_rejected = True
        return document


class ApplyDiscard(Filter):
    """
    このクラスは hojihca.filters.cleaners.ApplyDiscard に移動しました.

    上流フィルタで破棄された`Document`を空文字列にします.

    `Document.is_rejected=True` の ドキュメントは無視されるため,
    このフィルタを `Compose` のコンストラクタに渡しても動作しません.
    このフィルタは主に`Compose` 内部や, `discard_filtered=False` を指定
    したデバッグ時などに利用されます.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            "hojichar.filters.document_filter.ApplyDiscard は廃止されます. \
        hojihca.filters.cleaners.ApplyDiscard に移行しました",
        )

    def apply(self, document: Document) -> Document:
        """
        >>> ApplyDiscard().apply(Document(text="hello", is_rejected=True)).text
        ''
        """
        if document.is_rejected:
            document.text = ""

        return document


class SentenceTokenizer(Filter):
    """
    このクラスは hojihca.filters.tokenization.SentenceTokenizer に移動しました.

    日本語を想定したセンテンス単位のトーカナイザです.
    句点`。`で文章を区切ります. これだけでは実際の日本語テキストで不十分な例が多くある
    (句点にピリオドが用いられる, 会話のカギカッコ内で句点が用いられるなど)ため,
    将来的には適切なセンテンス単位のトーカナイザに置き換えられるべきです.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            "hojichar.filters.document_filter.SentenceTokenizer は廃止されます. \
        hojihca.filters.tokenization.SentenceTokenizer に移行しました",
        )

    def apply(self, document: Document) -> Document:
        tokens = self.tokenize(document.text)
        document.set_tokens(tokens)
        return document

    def tokenize(self, text) -> List[str]:
        """
        >>> SentenceTokenizer().tokenize("おはよう。おやすみ。ありがとう。さよなら。")
        ['おはよう。', 'おやすみ。', 'ありがとう。', 'さよなら。']

        >>> SentenceTokenizer().tokenize("さよなら。また来週")
        ['さよなら。', 'また来週']
        """
        tokens = text.split("。")
        if len(tokens) > 1:
            if text.endswith("。"):
                tokens = [token + "。" for token in tokens[:-1]]
            else:
                last = tokens[-1]
                tokens = [token + "。" for token in tokens]
                tokens[-1] = last

        return tokens


class DocumentNormalizer(Filter):
    """
    このクラスは hojichar.filters.normalization.DocumentNormalizer に移動しました.

    正規化をします.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            "hojichar.filters.document_filter.DocumentNormalizer は廃止されます. \
        hojihca.filters.normalization.DocumentNormalizer に移行しました",
        )

    def apply(self, document: Document) -> Document:
        document.text = unicodedata.normalize("NFKC", document.text)
        return document


class JSONLoader(Filter):
    """
    テキストを Json として解釈し, `key` で指定した要素を文字列として
    doument に格納します.デフォルトの `key` は 'text' です.

    Json の読み込み, あるいは `key` の読み込みに失敗した際には例外を送出します.
    これらを無視する場合は, `ignore=True` にします. その際, 読み込みに失敗
    したドキュメントは破棄されます.
    """

    def __init__(self, key="text", ignore=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.key = key
        self.ignore = ignore

    def apply(self, document: Document) -> Document:
        """
        >>> JSONLoader()( '{"text": "hello, world", "words": 2}' )
        'hello, world'

        >>> JSONLoader()( '{"text": hello, world ....' ) # Broken JSON
        Traceback (most recent call last):
            ...
        json.decoder.JSONDecodeError: Expecting value: line 1 column 10 (char 9)

        >>> JSONLoader()( '{"words": 2}' )
        Traceback (most recent call last):
            ...
        KeyError: 'text'

        >>> JSONLoader(ignore=True).apply(Document('{"text": hello, world ....' )).is_rejected
        True
        """
        try:
            data = json.loads(document.text)
            document.text = str(data[self.key])
        except Exception as e:
            if self.ignore:
                document.is_rejected = True
                return document
            else:
                raise e

        return document


class DocumentLengthFilter(Filter):
    """
    `min_doc_len`, `max_doc_len` で指定した上限・下限の範囲内にないドキュメントを破棄します.
    デフォルトでは 200字 以上 50000字以内のテキストが受理されます.
    """

    def __init__(self, min_doc_len=None, max_doc_len=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            "hojichar.filters.document_filter.DocumentLengthFilter は廃止されます. \
        hojihca.filters.cleaners.DocumentLengthFilter に移行しました",
        )

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
        warnings.warn(
            "hojichar.filters.document_filter.NgWordsFilterJa は廃止されます. \
        hojihca.filters.cleaners.NgWordsFilterJa に移行しました",
        )

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
        warnings.warn(
            "hojichar.filters.document_filter.NgWordsFilterEn は廃止されます. \
        hojihca.filters.cleaners.NgWordsFilterEn に移行しました",
        )

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
        warnings.warn(
            "hojichar.filters.document_filter.DiscardAdultContentJa は廃止されます. \
        hojihca.filters.cleaners.DiscardAdultContentJa に移行しました",
        )

    def apply(self, doc: Document) -> Document:
        """
        >>> DiscardAdultContentJa().apply(Document("<TEST_STRING_OF_ADULT_KEYWORD>")).is_rejected
        True

        >>> DiscardAdultContentJa().apply(Document("ほうじ茶")).is_rejected
        False

        挙動は正しいが誤検知しているケース. 他にも, サック in リュックサック,
        >>> DiscardAdultContentJa().apply(Document("アスパラガス")).is_rejected  # Matching with NG keyword "アス"
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
        warnings.warn(
            "hojichar.filters.document_filter.DiscardAdultContentEn は廃止されます. \
        hojihca.filters.cleaners.DiscardAdultContentEn に移行しました",
        )

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
        warnings.warn(
            "hojichar.filters.document_filter.DiscardDiscriminationContentJa は廃止されます. \
        hojihca.filters.cleaners.DiscardDiscriminationContentJa に移行しました",
        )

    def apply(self, doc: Document) -> Document:
        """
        >>> DiscardDiscriminationContentJa().apply(Document("<TEST_STRING_OF_DISCRIMINATION_KEYWORD>")).is_rejected
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
        warnings.warn(
            "hojichar.filters.document_filter.DiscardViolenceContentJa は廃止されます. \
        hojihca.filters.cleaners.DiscardViolenceContentJa に移行しました",
        )

    def apply(self, doc: Document) -> Document:
        """
        >>> DiscardViolenceContentJa().apply(Document("<TEST_STRING_OF_VIOLENCE_KEYWORD>")).is_rejected
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
        warnings.warn(
            "hojichar.filters.document_filter.DiscardBBSComments は廃止されます. \
        hojihca.filters.cleaners.DiscardBBSComments に移行しました",
        )

        self.max_allowed_num = max_allowed_num
        self.keyword_pat = re.compile(
            r"\d{4}[年\.\-\/][\ ]*\d{1,2}[月\.\-\/][\ ]*\d{1,2}[日]*|コメント|SOLD OUT|レビュー|投稿|ページ|\([月火水木金土日]\)|質問|\d+話|楽天市場|-"
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
        warnings.warn(
            "hojichar.filters.document_filter.DiscardAds は廃止されます. \
        hojihca.filters.cleaners.DiscardAds に移行しました",
        )

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
        warnings.warn(
            "hojichar.filters.document_filter.AcceptJapanese は廃止されます. \
        hojihca.filters.cleaners.AcceptJapanese に移行しました",
        )

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
        warnings.warn(
            "hojichar.filters.document_filter.DiscardRareKuten は廃止されます. \
        hojihca.filters.cleaners.DiscardRareKuten に移行しました",
        )

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
        warnings.warn(
            "hojichar.filters.document_filter.HeaderFooterTagsRemover は廃止されます. \
        hojihca.filters.cleaners.HeaderFooterTagsRemover に移行しました",
        )

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
        Original legacy code removed a pattern r"« _ | Main | _ »" .
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
        warnings.warn(
            "hojichar.filters.document_filter.MaskPersonalInformation は廃止されます. \
        hojihca.filters.cleaners.MaskPersonalInformation に移行しました",
        )

        self.phone_pat = re.compile(
            r"((0|\+\d{1,3}[- ]?)(\d{2}[- ]?\d{4}[- ]?|\d[- ]?\d{4}[- ]?|\d{2}[- ]?\d{3}[- ]?|\d{3}[- ]?\d{2}[- ]?|\d{4}[- ]?\d{1}[- ]?))\d{4}"
        )
        self.email_pat = re.compile(
            r"[a-zA-Z0-9!#$%&'*+\-/=?^_`{|}~.]+@[A-Za-z0-9!#$%&'*+\-/=?^_`{|}~.]+(\.[A-Za-z0-9\-]+)"
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


class GenerateDedupLSH(Filter):
    """
    ドキュメントの重複判定に使用可能なハッシュ値を生成します。
    ハッシュ値は20個生成され、類似する(≒編集距離が近い)文章どうしのハッシュが似る性質を持ちます(Locality Sensitive Hashing)。
    2つの文書間で少なくとも1つのハッシュ値が一致する場合に重複として判定することができます。
    生成されたハッシュは `Document.dedup_lsh` 属性に文字列のリストとして保存されます。
    重複処理を実施する場合は、本フィルタを `hojichar.filters.document_filters.LSHDeduplicator` の前に適用します。
    """

    N_MINHASH = 200
    N_GRAM = 5
    N_BUCKETS = 20
    BUCKET_SIZE = 10

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        warnings.warn(
            "hojichar.filters.document_filter.GenerateDedupLSH は廃止されます. \
        hojihca.filters.deduplication.GenerateDedupLSH に移行しました",
        )

    @staticmethod
    def n_gram_tokenize(text: str, n: int) -> List[str]:
        """
        Tokenize a string into n-gram tokens.

        >>> GenerateDedupLSH().n_gram_tokenize("おはようございます。", 3)
        ['おはよ', 'はよう', 'ようご', 'うござ', 'ござい', 'ざいま', 'います', 'ます。']

        >>> GenerateDedupLSH().n_gram_tokenize("おはよ", 5)
        ['おはよ']
        """
        if len(text) < n:
            return [text]
        else:
            return [text[idx : idx + n] for idx in range(len(text) - n + 1)]

    @staticmethod
    def get_minhash(tokens: List[str], h: Callable[[str], int]) -> int:
        return min([h(text) for text in tokens])

    @staticmethod
    def hashfunc_signed_32_from_seed(seed: int) -> Callable[[str], int]:
        return lambda text: mmh3.hash(text, seed, signed=True)

    def calc_lsh(self, text) -> List[str]:
        fingerprints = []
        for seed in range(self.N_MINHASH):
            hashfunc = self.hashfunc_signed_32_from_seed(seed)
            minhash = self.get_minhash(self.n_gram_tokenize(text, n=self.N_GRAM), hashfunc)
            fingerprints.append(minhash)

        lshs = []
        for b in range(self.N_BUCKETS):
            lshs.append(
                str(b)
                + "+"
                + "".join(
                    [
                        format(fingerprints[r] & 0xFFFFFFFF, "08x")[-4:]
                        for r in range(b * self.BUCKET_SIZE, (b + 1) * self.BUCKET_SIZE)
                    ]
                )
            )

        return lshs

    def apply(self, doc: Document) -> Document:
        """
        編集距離の近い文書ではハッシュが類似します。次の例では、5番目のハッシュは完全一致し、`LSHDeduplicator` で重複と判定されます。
        >>> from pprint import pprint
        >>> d1 = Document("吾輩は猫である。名前はまだ無い。どこで生まれたかとんと見当がつかぬ。")
        >>> d2 = Document("吾輩は鳥である。名前はまだ無い。どこで生まれたかとんと見当がつかぬ。")

        >>> d1 = GenerateDedupLSH().apply(d1)
        >>> pprint(d1.dedup_lsh)
        ['0+07ad0b7b163f434643387f3f4799a2d466bccd0c',
         '1+7369a7be65e9f41aa5088d4cebc7cbd4a506d702',
         '2+f62bc7aca49ef1ab0765c99d83e0ab5c4bab55b4',
         '3+a14eb2369a28d2effc886447c5c43cefd1dd4c99',
         '4+05598bc382a7b67d7740d416585f69c3c82ed7db',
         '5+4bcfd982910ba7034476fd9ba1cf04b1688265e1',
         '6+5a0d9f2c4ae08e5a695f279e042ccb1679f7fc95',
         '7+f810d30d918fb4b0a7df4dcc7fa76ab723250ebc',
         '8+8da61607a3de16c10f3e70498211990973c255c6',
         '9+c38f4fb17626f9cbbc4470424f3a3528a471a7c8',
         '10+b2b18747bae648b8f0ea62eed439605aea50c22c',
         '11+948f27853abed61d2e18e99e546975d28f74b7a9',
         '12+1c61e34046c00a32652af1a9532013cb43de6ab6',
         '13+610cf5dd9cd7bc25c92eb43a2238827c4313052d',
         '14+7fc7bbcb7d94bc531749d237d4c4d413bf1c3885',
         '15+f945ded8493498677c52c36509d879c6b0f5b765',
         '16+acbff8ad0f7bc356b953c95a4876dd5d4fd547d0',
         '17+c9c0263cfb4217ccf246a4d39e88fbad022a89b2',
         '18+cb5f7a0e4182922de50f742beb36da5d2b7f5c42',
         '19+b4557e514254043bebfe94925b563ecf79e36100']

        >>> d2 = GenerateDedupLSH().apply(d2)
        >>> pprint(d2.dedup_lsh)
        ['0+07ad0b7b163f434608967f3f4799a2d466bccd0c',
         '1+7369d2c065e9f41a27ec8d4cebc7cbd4a50608a8',
         '2+d31ac7aca49e4e7a0765c99d6335ab5c4bab55b4',
         '3+40bcb2369a28d2effc88fe8e01b13cefd1dd4c99',
         '4+05598bc382a72b497740d416585f69c3d702d7db',
         '5+4bcfd982910ba7034476fd9ba1cf04b1688265e1',
         '6+5a0d4eb74ae08e5a695f279e0f44cb1679f7fc95',
         '7+f810d30d918fb4b0a2604dcc7fa76ab723250ebc',
         '8+8da6c577a3de16c10f3e70498211053b73c255c6',
         '9+c38f4fb17626f9cbbc4470424f3a3528a471a7c8',
         '10+ac8e8747e42d48b820c262ee066ebedeea50c22c',
         '11+948f27853abed61dbc307e28cce275d200d7b7a9',
         '12+1c613c22f78e39fbbfedf1a95320e8c543de6ab6',
         '13+5020f5dd9cd7bc25c92e4a542238827c4313052d',
         '14+e4aebbcb7d94bc533eced237d4c4d413bf1c3885',
         '15+f9451c70493498677c5276113a9400906207b765',
         '16+acbff8ad8a12ba10b953f42d35dfdd5d4fd547d0',
         '17+c9c0263cfb427852f246a4d39e88fbadf4d889b2',
         '18+36780a524182439de50f742bd3e9da5d2b7f22b8',
         '19+b4557e51e775eeb64650949276973ecf79e3a451']


        全く異なる文書に対しては、いずれも全く異なるハッシュを返します。
        >>> d3 = Document("祇園精舎の鐘の声、諸行無常の響きあり。")
        >>> d3 = GenerateDedupLSH().apply(d3)
        >>> pprint(d3.dedup_lsh)
        ['0+6e37fad4c02c4e46f408fe2f1c858d3f848932a9',
         '1+14134eb5f17113a1eebdd2d14913b29661d91caa',
         '2+e8ff733fe01200bc8115d1b2fea8af17659797b6',
         '3+7780500e8dbf9658ae3073fc59fbf38170bd221d',
         '4+58e8071379406406bc159482d9d55c6ef3e407c5',
         '5+fceac9bca8ec3f137ee165977ad16cc26f84b0ff',
         '6+ca69466d7ba1d4df65b78aca2a48c0477d4191f4',
         '7+bc368e0c2dd50adb80802a3968259f5d50c864ee',
         '8+58afee22cdbed89dec223b2689fb46f81e4b3fc4',
         '9+0e13a8555203fa6c8c7c15fb3cbcddc338bacad8',
         '10+6b8d0c47d429ccd01c5c3ac76f2d2c001c40e1b1',
         '11+216137f870a95ed51cd116e009faa68f3520372f',
         '12+dedd4a4ba549f3c5655d197a73e594525eea44fb',
         '13+21b499a6debcbf8c339c6e9b390aa5ebfc866ec8',
         '14+32a82d880e2677cc06d32f4a0309220e299e1ad8',
         '15+e2afc748a373db953fd26e07cc5ac66adfbe7f4b',
         '16+5cc6a70dcccb97aaeb7dc0d26bb42f7610c0e535',
         '17+316905fb4aa3a81140e0b387b55fdef29e8dc2a2',
         '18+42277ad49a2c2b803909d8c34cad688ea5b04fb7',
         '19+53a44ce086b43d32e06c3f66233d566a5927793e']
        """
        lshs = self.calc_lsh(doc.text)
        doc.dedup_lsh = lshs
        return doc


class LSHDeduplicator(Filter):
    """
    `hojichar.filters.document_filter.GenerateDedupLSH` で生成したハッシュ値を基に重複判定をします。
    対象コーパスが約 10^6 以下 (〜数十GBが目安) であれば、事前処理なしで重複処理が可能です。
    事前処理なしで(オンラインに)重複除去を行う場合、`online_dedup` フラグを `True` にします。`online_dedup` のデフォルト値は `True` です。

    より大きなコーパスではすべての文書のハッシュ値をメモリに載せることが困難になります。重複文書は文書全体の数パーセントだと考えられるため、重複文書の
    ハッシュ値のみをブラックリストとしてファイルから読み込むことで数百GBのコーパスに対して処理ができます。
    `blacklist_path` に指定したパスのファイルから重複ハッシュ値を読み込みます。ブラックリストのファイルは1行おきに
    `GenerateDedupLSH`で生成したハッシュ値が記録されていることを仮定します。
    `store_blacklist` フラグを `True` にすると、重複のあったハッシュ値を `LSHDeduplicator.blacklist` 属性に 文字列の集合として記録します。
    このオプションは、ブラックリストのハッシュ値のファイルの作成時などに有効です。`store_blacklist` フラグのデフォルト値は `False`です。
    """

    def __init__(
        self,
        online_dedup=True,
        blacklist_path="",
        store_blacklist=False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        warnings.warn(
            "hojichar.filters.document_filter.LSHDeduplicator は廃止されます. \
        hojihca.filters.deduplication.LSHDeduplicator に移行しました",
        )

        self.online_dedup = online_dedup
        self.store_blacklist = store_blacklist
        self.seen = set()
        if blacklist_path:
            with open(blacklist_path) as fp:
                for line in fp:
                    lsh = line.strip()
                    self.seen.add(lsh)

        if store_blacklist:
            self.blacklist = copy.copy(self.seen)

    def apply(self, doc: Document) -> Document:
        """
        >>> d1 = GenerateDedupLSH().apply(Document("Hello, World."))
        >>> d2 = GenerateDedupLSH().apply(Document("吾輩は猫である。名前はまだ無い。どこで生まれたかとんと見当がつかぬ。"))
        >>> d3 = GenerateDedupLSH().apply(Document("吾輩は鳥である。名前はまだ無い。どこで生まれたかとんと見当がつかぬ。"))
        >>> deduplicator = LSHDeduplicator()
        >>> deduplicator.apply(d1).is_rejected
        False
        >>> deduplicator.apply(d2).is_rejected
        False
        >>> deduplicator.apply(d3).is_rejected
        True
        """
        lshs = doc.dedup_lsh
        if len(lshs) == 0:
            assert ValueError(
                "LSHs for deduplication are not caluculated. Filter `GenerateDedupLSH` must be composed before this filter."
            )

        for lsh in lshs:
            if lsh in self.seen:
                doc.is_rejected = True
                if self.store_blacklist:
                    self.blacklist.add(lsh)

            if self.online_dedup:
                self.seen.add(lsh)

        return doc


if __name__ == "__main__":
    import doctest

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s]%(name)s:%(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.DEBUG)

    doctest.testmod()
