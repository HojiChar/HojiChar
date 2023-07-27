import json
import logging
import pathlib
import re
import time
import unicodedata
from os import PathLike
from typing import Any, Optional, Union

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

    def __init__(self, key: str = "text", ignore: bool = False, *args: Any, **kwargs: Any) -> None:
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

    def apply(self, document: Document) -> Document:
        """
        >>> JSONDumper()("hojichar")
        '{"text": "hojichar"}'
        """
        text = document.text
        if self.dump_reason:
            document.text = json.dumps(
                {
                    "text": text,
                    "is_rejected": document.is_rejected,
                    "reason": document.reject_reason,
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
    正規表現 "BBS Patern" に `max_allow_num` 回よりたくさんマッチする文書を破棄します.
    `max_allow_num` のデフォルト値は14です.
    正規表現 "BBS Patern" は下記のリンクで検証可能です.
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
