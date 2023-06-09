"""
文書の(近似)重複処理のためのモジュール.
"""
import copy
from os import PathLike
from typing import Any, Callable, List, Union

import mmh3

from hojichar.core.filter_interface import Filter
from hojichar.core.models import Document


class GenerateDedupLSH(Filter):
    """
    ドキュメントの重複判定に使用可能なハッシュ値を生成します。
    ハッシュ値は20個生成され、類似する(≒編集距離が近い)文章どうしのハッシュが似る性質を持ちます(Locality Sensitive Hashing)。
    2つの文書間で少なくとも1つのハッシュ値が一致する場合に重複として判定することができます。
    生成されたハッシュは `Document.dedup_lsh` 属性に文字列のリストとして保存されます。
    重複処理を実施する場合は、本フィルタを `hojichar.filters.deduplication.LSHDeduplicator` の前に適用します。
    """

    def __init__(
        self,
        n_minhash: int = 200,
        n_gram: int = 5,
        n_buckets: int = 20,
        bucket_size: int = 10,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert n_minhash == n_buckets * bucket_size
        self.N_MINHASH = n_minhash
        self.N_GRAM = n_gram
        self.N_BUCKETS = n_buckets
        self.BUCKET_SIZE = bucket_size

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
    def hashfunc_signed_32_from_seed(seed: int) -> Callable[[str], int]:
        return lambda text: mmh3.hash(text, seed, signed=True)

    @staticmethod
    def get_minhash(tokens: List[str], hashfunc: Callable[[str], int]) -> int:
        return min([hashfunc(text) for text in tokens])

    def calc_lsh(self, text: str) -> List[str]:
        """
        テキストから重複処理のためのハッシュ値のシーケンスを生成します.
        2文書間で, 少なくとも1つ同じハッシュ値を保つ場合, 文書は重複と判断します.
        (参考文献)
        https://arxiv.org/abs/2107.06499 , Appendix A
        http://infolab.stanford.edu/~ullman/mmds/ch3n.pdf Chapter 3.3

        Args:
            text (str): 入力テキスト

        Returns:
            List[str]: ハッシュ値のリスト, 各ハッシュ値は
            '0+07ad0b7b163f434643387f3f4799a2d466bccd0c' のような形式で,
            先頭2文字は何番目のハッシュ値かを表す.
            これにより, 重複処理ハッシュを一つのハッシュテーブルにプールすることで重複処理ができる.
        """
        # N_MINHASH 個の mmh3 ハッシュ値 から最終的に N_BUKET 個の重複処理ハッシュを計算する
        fingerprints = []
        for seed in range(self.N_MINHASH):
            hashfunc = self.hashfunc_signed_32_from_seed(seed)
            minhash = self.get_minhash(self.n_gram_tokenize(text, n=self.N_GRAM), hashfunc)
            fingerprints.append(minhash)

        # 速度のためにリスト内包で書いており, 可読性低め
        # 各 fingerprint 16進数表記にして, 下四桁をバケットサイズ個ずつ連結している
        # TODO Python だとオーバーヘッドが大きいので, C++ で実装しなおす
        lshs = []
        for bucket_idx in range(self.N_BUCKETS):
            lshs.append(
                str(bucket_idx)
                + "+"
                + "".join(
                    [
                        format(fingerprints[fp_idx], "04x")[-4:]  # 下四桁をtrim
                        for fp_idx in range(
                            bucket_idx * self.BUCKET_SIZE, (bucket_idx + 1) * self.BUCKET_SIZE
                        )
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
        ['0+f853f485e9c1bcbabcc880c1b8675d2c994432f4',
         '1+8c9758429a170be65af872b41439342c5afa28fe',
         '2+09d538545b620e55f89b36637c2054a4b455aa4c',
         '3+5eb24dca65d82d1103789bb93a3cc3112e23b367',
         '4+faa7743d7d59498388c02beaa7a1963d37d22825',
         '5+b431267e6ef558fdbb8a02655e31fb4f977e9a1f',
         '6+a5f360d4b52071a696a1d862fbd434ea8609036b',
         '7+07f02cf36e714b505821b23480599549dcdbf144',
         '8+725ae9f95c22e93ff0c28fb77def66f78c3eaa3a',
         '9+3c71b04f89da063543bc8fbeb0c6cad85b8f5838',
         '10+4d4f78b9451ab7480f169d122bc79fa615b03dd4',
         '11+6b71d87bc54229e3d1e81662ab978a2e708c4857',
         '12+e39f1cc0b940f5ce9ad60e57ace0ec35bc22954a',
         '13+9ef40a23632943db36d24bc6ddc87d84bcedfad3',
         '14+80394435826c43ade8b72dc92b3c2bed40e4c77b',
         '15+06bb2128b6cc679983ae3c9bf628863a4f0b489b',
         '16+53410753f0853caa46ad36a6b78a22a3b02bb830',
         '17+3640d9c404bee8340dba5b2d61780453fdd6764e',
         '18+34a185f2be7e6dd31af18bd514ca25a3d481a3be',
         '19+4bab81afbdacfbc514026b6ea4aac131861d9f00']

        >>> d2 = GenerateDedupLSH().apply(d2)
        >>> pprint(d2.dedup_lsh)
        ['0+f853f485e9c1bcbaf76a80c1b8675d2c994432f4',
         '1+8c972d409a170be6d81472b41439342c5afaf758',
         '2+2ce638545b62b186f89b36639ccb54a4b455aa4c',
         '3+bf444dca65d82d1103780172fe4fc3112e23b367',
         '4+faa7743d7d59d4b788c02beaa7a1963d28fe2825',
         '5+b431267e6ef558fdbb8a02655e31fb4f977e9a1f',
         '6+a5f3b149b52071a696a1d862f0bc34ea8609036b',
         '7+07f02cf36e714b505da0b23480599549dcdbf144',
         '8+725a3a895c22e93ff0c28fb77deffac58c3eaa3a',
         '9+3c71b04f89da063543bc8fbeb0c6cad85b8f5838',
         '10+537278b91bd3b748df3e9d12f992412215b03dd4',
         '11+6b71d87bc54229e343d081d8331e8a2eff294857',
         '12+e39fc3de0872c60540130e57ace0173bbc22954a',
         '13+afe00a23632943db36d2b5acddc87d84bcedfad3',
         '14+1b524435826c43adc1322dc92b3c2bed40e4c77b',
         '15+06bbe390b6cc679983ae89efc56cff709df9489b',
         '16+5341075375ee45f046ad0bd3ca2122a3b02bb830',
         '17+3640d9c404be87ae0dba5b2d617804530b28764e',
         '18+c988f5aebe7ebc631af18bd52c1725a3d481dd48',
         '19+4bab81af188b114ab9b06b6e8969c131861d5baf']


        全く異なる文書に対しては、いずれも全く異なるハッシュを返します。
        >>> d3 = Document("祇園精舎の鐘の声、諸行無常の響きあり。")
        >>> d3 = GenerateDedupLSH().apply(d3)
        >>> pprint(d3.dedup_lsh)
        ['0+91c9052c3fd4b1ba0bf801d1e37b72c17b77cd57',
         '1+ebedb14b0e8fec5f11432d2fb6ed4d6a9e27e356',
         '2+17018cc11feeff447eeb2e4e015850e99a69684a',
         '3+8880aff2724169a851d08c04a6050c7f8f43dde3',
         '4+a718f8ed86c09bfa43eb6b7e262ba3920c1cf83b',
         '5+031636445714c0ed811f9a69852f933e907c4f01',
         '6+3597b993845f2b219a497536d5b83fb982bf6e0c',
         '7+43ca71f4d22bf5257f80d5c797db60a3af389b12',
         '8+a75111de3242276313dec4da7605b908e1b5c03c',
         '9+f1ed57abadfd05947384ea05c344223dc7463528',
         '10+9473f3b92bd73330e3a4c53990d3d400e3c01e4f',
         '11+de9fc8088f57a12be32fe920f6065971cae0c8d1',
         '12+2123b5b55ab70c3b9aa3e6868c1b6baea116bb05',
         '13+de4c665a21444074cc649165c6f65a15037a9138',
         '14+cd58d278f1da8834f92dd0b6fcf7ddf2d662e528',
         '15+1d5138b85c8d246bc02e91f933a63996204280b5',
         '16+a33a58f33335685614833f2e944cd08aef401acb',
         '17+ce97fa05b55d57efbf204c794aa1210e61733d5e',
         '18+bdd9852c65d4d480c6f7273db35397725a50b049',
         '19+ac5cb320794cc2ce1f94c09adcc3a996a6d986c2']
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
        online_dedup: bool = True,
        blacklist_path: Union[str, PathLike] = "",
        store_blacklist: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
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
                "LSHs for deduplication are not caluculated. Filter \
                    `GenerateDedupLSH` must be composed before this filter."
            )

        for lsh in lshs:
            if lsh in self.seen:
                doc.is_rejected = True
                if self.store_blacklist:
                    self.blacklist.add(lsh)

            if self.online_dedup:
                self.seen.add(lsh)

        return doc
