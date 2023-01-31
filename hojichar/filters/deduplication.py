"""
文書の(近似)重複処理のためのモジュール.
"""
import copy
import logging
from typing import Callable, List

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

    N_MINHASH = 200
    N_GRAM = 5
    N_BUCKETS = 20
    BUCKET_SIZE = 10

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

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
        # 各 fingerprint を 符号あり -> 符号なしにして, 16進数表記にして, 下四桁をバケットサイズ個ずつ連結している
        # HACK 符号なしにする操作は不要, 既存のハッシュ値との互換性を破壊して再実装する際は取り除いてよい
        # TODO Python だとオーバーヘッドが大きいので, C++ で実装しなおす
        lshs = []
        for bucket_idx in range(self.N_BUCKETS):
            lshs.append(
                str(bucket_idx)
                + "+"
                + "".join(
                    [
                        format(fingerprints[fp_idx] & 0xFFFFFFFF, "08x")[-4:]
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


if __name__ == "__main__":
    import doctest

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s]%(name)s:%(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.DEBUG)

    doctest.testmod()
