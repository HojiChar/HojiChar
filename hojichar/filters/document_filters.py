import json
import logging
import pathlib
import re
import string
import time
import unicodedata
from collections import Counter
from itertools import groupby
from os import PathLike
from typing import Any, Dict, List, Optional, Union

import numpy as np

import hojichar
from hojichar.core.filter_interface import Filter
from hojichar.core.models import Document, Token

try:
    import emoji
    from fugashi import Tagger  # type: ignore

    is_loaded_extras = True
except ImportError:
    is_loaded_extras = False

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

    def __init__(self, keyword: str, *args: Any, **kwargs: Any) -> None:
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


class Identity(Filter):
    """何も変化を加えないフィルタです. テスト・デバッグに用いられます."""

    def apply(self, document: Document) -> Document:
        return document


class DiscardAll(Filter):
    """
    すべてのドキュメントを破棄するフィルタです.
    テスト・デバッグに用いられます.
    """

    def apply(self, document: Document) -> Document:
        document.is_rejected = True
        return document


class ApplyDiscard(Filter):
    """
    上流フィルタで破棄された`Document`を空文字列にします.

    `Document.is_rejected=True` の ドキュメントは無視されるため,
    このフィルタを `Compose` のコンストラクタに渡しても動作しません.
    このフィルタは主に`Compose` 内部や, `discard_filtered=False` を指定
    したデバッグ時などに利用されます.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def apply(self, document: Document) -> Document:
        """
        >>> ApplyDiscard().apply(Document(text="hello", is_rejected=True)).text
        ''
        """
        if document.is_rejected:
            document.text = ""

        return document


class Sleep(Filter):
    """
    デバッグ用のフィルタです. 指定秒スリープします.
    """

    def __init__(self, time: float = 1.0, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.time = time

    def apply(self, document: Document) -> Document:
        """
        >>> Sleep(0.1)('hello')  # After 0.1 seconds,
        'hello'
        """
        time.sleep(self.time)
        return document


class DocumentNormalizer(Filter):
    """
    Unicode の正規化をします.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

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

    def __init__(
        self,
        key: str = "text",
        ignore: bool = False,
        extra_keys: Optional[List[str]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.key = key
        self.ignore = ignore
        self.extra_keys = extra_keys

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
            if self.extra_keys is not None:
                document.extras = {key: data[key] for key in self.extra_keys if key in data}
        except Exception as e:
            logger.error(f"Failed to parsing in JSONLoader. Input document: \n{document.text}")
            if self.ignore:
                document.is_rejected = True
                return document
            else:
                raise e

        return document


class JSONDumper(Filter):
    """
    Document.text の文字列を json に変換します.
    必要に応じ Document のメタデータを付与します. これはドキュメントの破棄事由が含まれ、偽陽性の分析に有効です。
    デフォルトで `skip_rejected` が `False` にセットされており、Document の破棄フラグにかかわらず
    処理されます。
    """

    def __init__(
        self,
        dump_reason: bool = False,
        p: float = 1,
        skip_rejected: bool = False,
        export_extras: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            dump_reason (bool, optional): `is_rejected`, `reason` エントリをダンプします. Defaults to False.
            p (float, optional): Apply probability. Defaults to 1.
            skip_rejected (bool, optional): 破棄済みサンプルを排除しません.
        """
        super().__init__(p, skip_rejected, *args, **kwargs)
        self.dump_reason = dump_reason
        self.export_extras = export_extras

    def apply(self, document: Document) -> Document:
        """
        >>> JSONDumper()("hojichar")
        '{"text": "hojichar"}'
        """
        text = document.text
        if self.dump_reason:
            if self.export_extras:
                document.text = json.dumps(
                    {
                        "text": text,
                        "is_rejected": document.is_rejected,
                        "reason": document.reject_reason,
                        "extras": document.extras,
                    },
                    ensure_ascii=False,
                )
            else:
                document.text = json.dumps(
                    {
                        "text": text,
                        "is_rejected": document.is_rejected,
                        "reason": document.reject_reason,
                    },
                    ensure_ascii=False,
                )
        else:
            if self.export_extras:
                document.text = json.dumps(
                    {
                        "text": text,
                        "extras": document.extras,
                    },
                    ensure_ascii=False,
                )
            else:
                document.text = json.dumps({"text": text}, ensure_ascii=False)
        return document


class DocumentLengthFilter(Filter):
    """
    `min_doc_len`, `max_doc_len` で指定した上限・下限の範囲内にないドキュメントを破棄します.
    デフォルトでは 200字 以上 50000字以内のテキストが受理されます.
    """

    def __init__(
        self,
        min_doc_len: Optional[int] = None,
        max_doc_len: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
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

    def __init__(
        self,
        dict_path: Union[str, PathLike],
        ignore_confused: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
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
        regex_match = self.keyword_pat.search(doc.text)
        if regex_match:
            doc.is_rejected = True
            self.matched_text = regex_match.group()
            self.matched_text_neighbor = doc.text[
                regex_match.start() - 20 : regex_match.end() + 20
            ]

        return doc


class NgWordsFilterEn(Filter):
    """
    英語のNGワード(および不適切語)を含む文書を破棄します.
    `dict_path` で指定したファイルから, キーワードのリストを得ます.
    ファイルは単語が改行で羅列されたテキストファイルです.
    """

    def __init__(self, dict_path: Union[str, PathLike], *args: Any, **kwargs: Any) -> None:
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

    def __init__(
        self,
        dict_path: Union[str, PathLike] = BASE_PATH / "dict/adult_keywords_ja.txt",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(dict_path, *args, **kwargs)

    def apply(self, doc: Document) -> Document:
        """
        >>> DiscardAdultContentJa().apply(Document("<TEST_STRING_OF_ADULT_KEYWORD>")).is_rejected
        True

        >>> DiscardAdultContentJa().apply(Document("ほうじ茶")).is_rejected
        False

        挙動は正しいが誤検知しているケース. 他にも, サック in リュックサック,
        >>> DiscardAdultContentJa().apply(Document("アスパラガス")).is_rejected \
        # Matching with NG keyword "アス"
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

    def __init__(
        self,
        dict_path: Union[str, PathLike] = BASE_PATH / "dict/adult_keywords_en.txt",
        *args: Any,
        **kwargs: Any,
    ) -> None:
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
        dict_path: Union[str, PathLike] = BASE_PATH / "dict/discrimination_keywords_ja.txt",
        *args: Any,
        **kwargs: Any,
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

    def __init__(
        self,
        dict_path: Union[str, PathLike] = BASE_PATH / "dict/violence_keywords_ja.txt",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(dict_path, *args, **kwargs)

    def apply(self, doc: Document) -> Document:
        """
        >>> DiscardViolenceContentJa()\
            .apply(Document("<TEST_STRING_OF_VIOLENCE_KEYWORD>")).is_rejected
        True

        >>> DiscardViolenceContentJa().apply(Document("ほうじ茶")).is_rejected
        False
        """
        return super().apply(doc)


class DiscardBBSComments(Filter):
    """
    正規表現 "BBS Pattern" に `max_allow_num` 回よりたくさんマッチする文書を破棄します.
    `max_allow_num` のデフォルト値は14です.
    正規表現 "BBS Pattern" は下記のリンクで検証可能です.
    https://regex101.com/r/ybQvL2/1
    """

    def __init__(self, max_allowed_num: int = 14, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.max_allowed_num = max_allowed_num
        self.keyword_pat = re.compile(
            r"\d{4}[年\.\-\/][\ ]*\d{1,2}[月\.\-\/][\ ]*\d{1,2}[日]*|コメント|SOLD OUT|レビュー|投稿|ページ|\([月火水木金土日]\)|質問|\d+話|楽天市場|-"  # noqa
        )

    def apply(self, doc: Document) -> Document:
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
    主に広告キーワードを`max_allow_num`より多く含む文書を破棄します.
    デフォルトで`max_allow_num` は14です.
    `dict_path` で指定したファイルから, 広告キーワードのリストを得ます.
    ファイルは単語が改行で羅列されたテキストファイルです.
    デフォルトの`dict_path` は /hojichar/dict/advertisement_keywords_ja.txt です.
    """

    def __init__(
        self,
        dict_path: Union[str, PathLike] = BASE_PATH / "dict/advertisement_keywords_ja.txt",
        max_allowed_num: int = 14,
        *args: Any,
        **kwargs: Any,
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

    def __init__(self, lookup_size: int = 50, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.lookup_size = lookup_size
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

    def __init__(self, max_average_sentence_length: int = 100, *args: Any, **kwargs: Any) -> None:
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
        dict_path: Union[str, PathLike] = BASE_PATH / "dict/header_footer_keywords_ja.txt",
        *args: Any,
        **kwargs: Any,
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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
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


class DiscardTooManyNouns(Filter):
    """
    [!CAUTION] This filter requires `fugashi` package. Please install it
    by `pip install 'hojichar[all]'`.

    A filter that removes document with too many nouns in Japanese i.e.,
    documents such as advertisement, word salad, etc ...
    """

    def __init__(self, threshold: float = 0.80, *args: Any, **kwargs: Any) -> None:
        """
        Args:
            threshold: document whose noun ratio is higher than this value will be discarded
            *args:
            **kwargs:
        """
        super().__init__(*args, **kwargs)
        assert (
            is_loaded_extras
        ), "fugashi is required for this filter. Try pip install 'hojichar[all]'"

        self.threshold = threshold
        self.tagger = Tagger("-Owakati")
        assert (
            "unidic" in self.tagger.dictionary_info[0]["filename"]
        ), "MeCab dictionary must be unidic"

    def apply(self, doc: Document) -> Document:
        """
        >>> DiscardTooManyNouns().apply(Document("自然言語処理大好き！")).is_rejected
        False
        >>> DiscardTooManyNouns().apply(Document("リンゴ・オレンジ・ミカン・バナナ セール中")).is_rejected
        True
        >>> DiscardTooManyNouns().apply(Document("今日の仙台朝市ではリンゴがセール中")).is_rejected
        False
        """
        # remove "補助記号" from part-of-speech statistics
        # because they often decrease the noun ratio,
        # e.g., the sentence "リンゴ・オレンジ・バナナ・" has 補助記号 ratio of 0.5
        # however, we don't want such sentence
        pos_count = Counter(
            w.feature.pos1 for w in self.tagger(doc.text) if w.feature.pos1 != "補助記号"
        )
        try:
            noun_ratio = pos_count["名詞"] / sum(pos_count.values())
        except ZeroDivisionError:
            noun_ratio = 0.0
        if noun_ratio >= self.threshold:
            doc.is_rejected = True
        return doc


class CharRepetitionRatioFilter(Filter):
    """
    文字Ngramの重なり率（文書中で高頻度文字Ngramが占める割合）を計算して, 重なりの大きいものを除去します.
    名詞の連続からなるような広告テキストを取り除くのに有効です.

    実装は, BigScience で採用されていた前処理を参考にしています.
    元実装: https://github.com/bigscience-workshop/data-preparation/blob/9d0588419073cc5bf0fb92b58f37f2a1016572c3/preprocessing/training/01b_oscar_cleaning_and_filtering/filtering.py#L425-L453  # noqa: E501

    「高頻度文字Ngram」は、sqrt(ユニークなNgramの総数)によって求めていますが,
    これは文書長の影響を軽減するためだとされています.

    掲示板のテキストが引っかかりやすい傾向があります.
    13: 名無しさん@実況で競馬板アウト 2019/08/18(日) 15:28:46.10 ID:eBvZg8h+0
    的なものが高頻度で登場するため、文字Ngramの重なり率も高くなってしまう
    """

    def __init__(
        self, threshold: float = 0.33, ngram_size: int = 5, *args: Any, **kwargs: Any
    ) -> None:
        """

        Args:
            threshold: document with character repetition ratio higher than this value will be discarded
            ngram_size: character ngram size. Larger value will decrease the false positive of long documents
            *args:
            **kwargs:
        """  # noqa: E501

        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.ngram_size = ngram_size

    def apply(self, doc: Document) -> Document:
        ratio = self.compute_character_repetition_ratio(doc.text, self.ngram_size)
        if ratio >= self.threshold:
            doc.is_rejected = True
        return doc

    @staticmethod
    def compute_character_repetition_ratio(
        document: str, character_repetition_length: int
    ) -> float:
        def get_freq_character_ngrams(document: str, n: int) -> Dict[str, int]:
            character_ngrams: List[str] = [
                document[i : i + n] for i in range(len(document) - n + 1)
            ]
            freq_character_ngrams_dict: Dict[str, int] = {}
            for character_ngram in character_ngrams:
                freq_character_ngrams_dict[character_ngram] = (
                    freq_character_ngrams_dict.get(character_ngram, 0) + 1
                )
            return freq_character_ngrams_dict

        freq_character_ngrams_dict = get_freq_character_ngrams(
            document, character_repetition_length
        )
        if len(freq_character_ngrams_dict) == 0:
            return 0.0
        freq_character_ngrams: List[int] = list(freq_character_ngrams_dict.values())
        freq_character_ngrams = sorted(freq_character_ngrams, reverse=True)
        val_one = len([el for el in freq_character_ngrams if el == 1])
        num_rep_character_ngrams = min(
            int(np.sqrt(len(freq_character_ngrams))),
            len(freq_character_ngrams) - val_one,
        )
        character_repetition_ratio = sum(freq_character_ngrams[:num_rep_character_ngrams]) / sum(
            freq_character_ngrams
        )
        return character_repetition_ratio


class WordRepetitionRatioFilter(Filter):
    """
    [!CAUTION] This filter requires `fugashi` package. Please install it
    by `pip install 'hojichar[all]'`.

    単語Ngramの重なり率（文書中で重複する単語Ngramが占める割合）を計算して、重なりの大きいものを弾くためのフィルタ.
    BigScienceで採用されていた前処理を参考にしている.

    名詞が連打されているような広告テキストを取り除くのに有効な様子
    まともな文書がたまたま2回繰り返されている場合もあり、これを取り除いて良いのかは分からない
    例：
    "ウェブ\n本文: ニコンの上昇率16%超える、今3月期は経常76%の大幅増益見込む(ニコン) 2013年05月10日[minkabu PRESS] - みんなの株式 (みんかぶ)\n2013/05/10(10:57)
    ニコン<7731.T>が急騰、寄り付き直後に前日比355円高の2537円まで買い上げ
    られ、上昇率は16%を超えた。外国為替市場で円が1ドル100円台、1ユーロ131円台に入るなど急速に円安が進み、輸出株が軒並み高になる
    なか、9日取引終了後に発表した前年3月期決算で、今3月期は2ケタ近い増収で大幅増益を見込んだことが買い気を強めさせた。連結売上
    高は前期比9.8%増の1兆1100億円、経常利益75.8%増の850億円を予想。前期は半導体、電子部品の低迷が足かせになり、2ケタ増収ながら
    経常46%の大幅減益になったが、レンズ交換式デジタルカメラの拡大や液晶ディスプレイの回復で収益が急回復する。ニコンの株価は10時
    56分現在2491円(△309円)出所:株経通信(株式会社みんかぶ)\n2013/05/10 - ニコン(7731) の関連ニュース。 ニコン<7731.T>が急騰、寄
    り付き直後に前日比355円高の2537円まで買い上げられ、上昇率は16%を超えた。外国為替市場で円が1ドル100円台、1ユーロ131円台に入
    るなど急速に円安が進み、輸出株が軒並み高になるなか、9日取引終了後に発表した前年3月期決算で、今3月期は2ケタ近い増収で大幅増
    益を見込んだことが買い気を強めさせた。連結売上高は前期比9.8%増の1兆1100億円、経常利益75.8%増の850億円を予想。前期は半導体、
    電子部品の低迷が足かせになり、2ケタ増収ながら経常46%の大幅減益になったが、レンズ交換式デジタルカメラの拡大や液晶ディスプレ
    イの回復で収益が急回"
    """  # noqa: E501

    def __init__(
        self, threshold: float = 0.40, ngram_size: int = 7, *args: Any, **kwargs: Any
    ) -> None:
        """

        Args:
            threshold: document whose character repetition ratio is higher than this value will be discarded
            ngram_size: character ngram size. Larger value will decrease the false positive of long documents
            *args:
            **kwargs:
        """  # noqa: E501
        super().__init__(*args, **kwargs)
        assert (
            is_loaded_extras
        ), "fugashi is required for this filter. Try pip install 'hojichar[all]'"

        self.threshold = threshold
        self.ngram_size = ngram_size
        self.tagger = Tagger("-Owakati")

    def apply(self, doc: Document) -> Document:
        ratio = self.compute_word_repetition_ratio(doc.text, self.ngram_size)
        if ratio >= self.threshold:
            doc.is_rejected = True
        return doc

    def compute_word_repetition_ratio(self, document: str, word_repetition_length: int) -> float:
        def get_freq_word_ngrams(document: str, n: int) -> Dict[str, int]:
            # tokenizing given document
            words = [w.surface for w in self.tagger(document)]
            word_ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
            freq_word_ngrams: Dict[str, int] = {}
            for word_ngram in word_ngrams:
                freq_word_ngrams[word_ngram] = freq_word_ngrams.get(word_ngram, 0) + 1
            return freq_word_ngrams

        freq_word_ngrams_dict = get_freq_word_ngrams(document, word_repetition_length)
        if len(freq_word_ngrams_dict) == 0:
            return 0
        freq_word_ngrams = list(freq_word_ngrams_dict.values())
        word_repetition_ratio = sum(freq for freq in freq_word_ngrams if freq > 1) / sum(
            freq_word_ngrams
        )

        return word_repetition_ratio


class DiscardTooManySpecialToken(Filter):
    """
    [!CAUTION] This filter requires `emoji` package. Please install it
    by `pip install 'hojichar[all]'`.

    句読点を含む記号、空白、絵文字、その他特殊な文字を一定の割合以上含むような文書を取り除くためのフィルタ
    元実装: BigScience https://github.com/bigscience-workshop/data-preparation/blob/9d0588419073cc5bf0fb92b58f37f2a1016572c3/preprocessing/training/01b_oscar_cleaning_and_filtering/parameters_filtering.py#L5-L16  # noqa: E501
    """

    def __init__(self, threshold: float = 0.4, *args: Any, **kwargs: Any) -> None:
        """

        Args:
            threshold: document whose special token ratio is higher than this value will be discarded
            *args:
            **kwargs:
        """  # noqa: E501
        super().__init__(*args, **kwargs)

        # digits are not regarded as special tokens
        # otherwise many false positives are made, i.e., good documents discarded
        main_special_characters = string.punctuation + string.whitespace  # + string.digits
        other_special_characters = (
            "    　    ￼’“”–▬…✦�­£​•€«»°·═"
            "×士＾˘⇓（）§″′´¿−±∈﻿¢ø‚„½¼¾¹²³―⁃，ˌ¸‹›ʺˈʻ¦‐⠀‰‑≤≥‖"
            "◆●■►▼▲▴∆▻¡★☆✱ːº。¯˜¥ɪ≈†：⁄♡✓⊕․．⋅÷１‟；،、¨ाাी्े◦˚"
            "゜ʼ≖ʼ¤℃√！？【】‿∞➤～πه۩☛₨➩☻๑٪♥ıॽ《‘©﴿٬？▷Г♫∟™ª₪®「—❖"
            "」﴾》�"
        )

        en_emoji = emoji.EMOJI_DATA.keys()

        special_characters_default = set(main_special_characters + other_special_characters)
        special_characters_default.update(en_emoji)
        self.special_characters = special_characters_default

        self.threshold = threshold

    def _compute_special_characters_ratio(self, text: str) -> float:
        if len(text) == 0:
            return 0

        special_characters_ratio = len(
            [char for char in text if char in self.special_characters]
        ) / len(text)
        return special_characters_ratio

    def apply(self, doc: Document) -> Document:
        special_characters_ratio = self._compute_special_characters_ratio(doc.text)

        if special_characters_ratio > self.threshold:
            doc.is_rejected = True
        return doc


class SingleCharacterRepetitionFilter(Filter):
    """
    単一文字が大量に繰り返されているような文書を取り除くためのフィルタ
    そのような文書はノイズである可能性が高いため
    参考: BigScienceプロジェクトによると、oscarデータセットの中にバックスラッシュだけを2M個含むような文書が含まれていたらしい
    https://github.com/bigscience-workshop/bigscience/blob/master/train/tr8-104B-wide/chronicles.md#2m-backslash-only-samples-in-our-dataset  # noqa: E501
    """

    def __init__(
        self,
        threshold: int = 200,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            threshold: The document is removed if character is repeated for this value or more
            *args:
            **kwargs:
        """
        super().__init__(*args, **kwargs)
        self.threshold = threshold

    def _is_repeat_contained(self, text: str) -> bool:
        groups = groupby(text)
        is_repeat_contained = any(sum(1 for _ in group) >= self.threshold for _, group in groups)
        return is_repeat_contained

    def apply(self, doc: Document) -> Document:
        if self._is_repeat_contained(doc.text):
            doc.is_rejected = True
        return doc


class DiscardTooManyEndingEllipsis(Filter):
    """
    ellipsisで終わるような行が大量に含まれるような文書を取り除くためのフィルタです.
    ellipsisとしては ... と … を用いている
    同様のフィルタが RedPajama v2で用いられています.

    例として, 以下のような文書を検知します.
    ```
    ペアーズは女性、という驚愕の過食が出ているのをごアラサーですか。時代から付...
    バツイチアラフォー 婚活ち女性の特徴と子持な付...
    ```

    デフォルトではしきい値を0.7としているが, これはC4から0.1%を削るような設定であり、
    precisionを重視した設定です.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            threshold: The document is removed if ratio of lines ending with ellipsis is higher than this value
            *args:
            **kwargs:
        """  # noqa: E501
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.ellipsis_pattern = re.compile(r"(\.{3}|…)\n")  # matches ...\n and …\n

    def apply(self, doc: Document) -> Document:
        ellipsis_count = len(self.ellipsis_pattern.findall(doc.text))
        newline_count = max(doc.text.count("\n"), 1)  # avoid zero division
        ellipsis_ratio = ellipsis_count / newline_count

        if ellipsis_ratio > self.threshold:
            doc.is_rejected = True
        return doc


class DiscardTooShortLines(Filter):
    """
    短い行を大量に含む文書を捨てるためのフィルタです.

    メニューバーやパンくずリストのような要素を大量に含む文書を取り除くのに有効です.
    """

    def __init__(self, threshold: float = 0.5, *args: Any, **kwargs: Any) -> None:
        """
        Args:
            threshold: The document is removed if the ratio of short (<10 chars) lines are more than this value.
            *args:
            **kwargs:
        """  # noqa: E501
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        # この値は適当に決め打ち
        self.minimum_line_length = 10

    def apply(self, doc: Document) -> Document:
        lines = [len(x) for x in doc.text.split("\n")]
        short_lines = [x for x in lines if x <= self.minimum_line_length]
        if (len(short_lines) / len(lines)) > self.threshold:
            doc.is_rejected = True
        return doc
