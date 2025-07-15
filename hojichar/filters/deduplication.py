from __future__ import annotations

import importlib
import re
from typing import Any, Callable, Iterable, Optional

import numpy as np
import rensa
import xxhash
from nltk.util import ngrams  # type: ignore
from numpy.random._generator import Generator as Generator
from scipy.integrate import quad as integrate  # type: ignore

from hojichar import AsyncFilter, Document, Filter

_japanese_tagger: Optional["fugashi.Tagger"] = None
NON_ALPHA = re.compile("[^A-Za-z_0-9]")


def non_alpha_num_splitter(text: str) -> list[str]:
    """
    Split the text into alphanumeric tokens.
    This is a simple implementation that splits on non-alphanumeric characters.
    """
    return [token for token in NON_ALPHA.split(text) if token]


def japanese_word_splitter(text: str) -> list[str]:
    """
    Split the text into Japanese words using fugashi.
    This will import fugashi and instantiate Tagger on first use.
    """
    global _japanese_tagger
    if _japanese_tagger is None:
        fugashi = importlib.import_module("fugashi")
        _japanese_tagger = fugashi.Tagger()
    return [token.surface for token in _japanese_tagger(text)]  # type: ignore


class GenerateDedupLSH(Filter):
    def __init__(
        self,
        num_perm: int = 500,
        threshold: float = 0.8,
        tokenizer: Callable[[str], Iterable[str]] = lambda x: list(x),
        n_grams: int = 5,
        seed: int = 42,
        inline_dedup: bool = True,
        **kwargs: Optional[dict],
    ):
        super().__init__(**kwargs)
        self.num_perm = num_perm
        self.threshold = threshold
        self.tokenizer = tokenizer
        self.n_grams = n_grams
        self.seed = seed

        self.num_bands, self.band_size = self.optimal_param(
            threshold=threshold,
            num_perm=num_perm,
            false_positive_weight=0.5,
            false_negative_weight=0.5,
        )

        self.inline_dedup = inline_dedup

    def calculate_minhash_signature(self, text: str) -> np.ndarray:
        """
        Calculate the MinHash signature for the given text.
        """
        tokens = self.tokenizer(text)
        n_gram_tokens = ngrams(tokens, self.n_grams)
        tokens = [" ".join(grams) for grams in n_gram_tokens]
        minhash = rensa.RMinHash(num_perm=self.num_perm, seed=self.seed)
        minhash.update(tokens)
        return np.asarray(minhash.digest(), dtype=np.uint32)

    def signature_to_lsh_key(self, signature: np.ndarray, band_size: int, band_idx: int) -> int:
        start = band_idx * band_size
        view = memoryview(signature[start : start + band_size])
        return xxhash.xxh128_intdigest(view)

    def apply(self, document: Document) -> Document:
        """
        Apply the deduplication filter to the document.
        """
        signature = self.calculate_minhash_signature(document.text)
        lsh_key = [
            f"{band_idx}+{self.signature_to_lsh_key(signature, self.band_size, band_idx):016x}"
            for band_idx in range(self.num_bands)
        ]
        document.extras["dedup_lsh"] = lsh_key
        return document

    @staticmethod
    def optimal_param(
        threshold: float,
        num_perm: int,
        false_positive_weight: float = 0.5,
        false_negative_weight: float = 0.5,
    ) -> tuple[int, int]:
        """
        Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
        of probabilities of false positive and false negative.
        This implementation is based on: https://github.com/ekzhu/datasketch/blob/5512549871d29c55b3cd8e99a79fcbd14859b77d/datasketch/lsh.py#L12-L40
        """

        def _false_positive_probability(threshold: float, b: int, r: int) -> float:
            _probability = lambda s: 1 - (1 - s ** float(r)) ** float(b)  # noqa
            a, err = integrate(_probability, 0.0, threshold)
            return a  # type: ignore

        def _false_negative_probability(threshold: float, b: int, r: int) -> float:
            _probability = lambda s: 1 - (1 - (1 - s ** float(r)) ** float(b))  # noqa
            a, err = integrate(_probability, threshold, 1.0)
            return a  # type: ignore

        min_error = float("inf")
        opt = (0, 0)
        for b in range(1, num_perm + 1):
            max_r = int(num_perm / b)
            for r in range(1, max_r + 1):
                fp = _false_positive_probability(threshold, b, r)
                fn = _false_negative_probability(threshold, b, r)
                error = fp * false_positive_weight + fn * false_negative_weight
                if error < min_error:
                    min_error = error
                    opt = (b, r)
        return opt


class InlineDeduplicator(Filter):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.hash_pool: set[str] = set()

    def apply(self, document: Document) -> Document:
        """
        Inline deduplication based on the LSH keys in the document.
        This filter cannot use in the distributed environment because it uses a local hash pool.
        """
        lsh_keys = document.extras.get("dedup_lsh")
        if lsh_keys is None:
            raise ValueError(
                "Document does not contain LSH keys for deduplication. Please apply GenerateDedupLSH first."
            )

        for lsh in lsh_keys:
            if lsh in self.hash_pool:
                document.is_rejected = True
            else:
                self.hash_pool.add(lsh)
        return document


if __name__ == "__main__":
    import argparse

    from hojichar import Compose
    from hojichar.filters.document_filters import JSONDumper, JSONLoader

    parser = argparse.ArgumentParser(description="Run NearDedup filter example.")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    args = parser.parse_args()
    # Example usage
    pipeline = Compose(
        [
            JSONLoader(),
            GenerateDedupLSH(
                num_perm=500, threshold=0.8, n_grams=5, tokenizer=non_alpha_num_splitter
            ),
            InlineDeduplicator(num_perm=500, threshold=0.8),
            JSONDumper(export_extras=True, dump_reason=True),
        ]
    )
    with open(args.input, "r") as f:
        docs = (Document(line) for line in f)
        out_iter = pipeline.apply_stream(docs)
        for doc in out_iter:
            print(doc.text)
