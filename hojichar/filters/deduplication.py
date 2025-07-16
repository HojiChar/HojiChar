"""
This module provides filters for deduplication using MinHash and Locality-Sensitive Hashing (LSH).
If you want to use this module, install hojichar via `pip install 'hojichar[dedup]'`

## What is MinHash LSH?
*A gentle introduction for first‑time leaner for MinHash LSH*

---

### 1 What problem does this solve?

When you have **millions of documents** it is too expensive to compare every pair directly.
**MinHash + Locality‑Sensitive Hashing (LSH)** lets you

- estimate Jaccard similarity extremely fast, and memory‑efficiently
- retrieve **near‑duplicates** in sub‑linear time.

In practice you can keep a single `set` or Redis index of the generated **LSH keys** and ask:
"Does any existing document share *at least one* LSH key with mine?"

If the answer is yes the two documents are almost certainly similar; if no they are very likely different.

### How the pipeline works

`GenerateDedupLSH` filter generates LSH keys for each document.

1. Tokenize
  - Split text into tokens. (Default: character‑level, but you can plug in any callable.)
2. n-grams
  - Group tokens into n‑grams (n_grams=5) to capture context.
3. MinHash
  - Hash each n‑gram with `num_perm` independent permutations and keep only the minimum value. The resulting **signature** is a vector of num_perm 32‑bit integers.
  - This module uses `rensa.RMinHash` which is a fast MinHash implementation by Rust language.
4. Banding and compression
  - Split the signature into b bands, each containing `r` integers (num_perm ≈ b×r).
  - Treat the `r` integers as raw bytes, hash them with xxhash‑128, and format as <band_idx>+<digest32hex>.
5. Output
  - Store all band keys in `document.extras['dedup_lsh']` as a list of strings.

`InlineDeduplicator`, `RedisDeduplicator`, and `RedisBloomDeduplicator` filters use the generated LSH keys to mark documents as duplicates.

- `InlineDeduplicator` stores LSH keys in a local set, so it works only in a single process.
- `RedisDeduplicator` stores LSH keys in Redis, so it works in a distributed environment.
- `RedisBloomDeduplicator` stores LSH keys in RedisBloom, which is a scalable Bloom filter. It uses less memory than Redis keys but may return false positives.

"""

from __future__ import annotations

import importlib
import re
from typing import Any, Callable, Final, Iterable, Optional

import numpy as np
import redis
import xxhash
from datasketch.lsh import _optimal_param  # type: ignore
from nltk.util import ngrams  # type: ignore
from numpy.typing import NDArray
from rensa import RMinHash  # type: ignore

from hojichar import Document, Filter

_japanese_tagger: Optional["fugashi.Tagger"] = None  # type: ignore[name-defined] # noqa: F821
NON_ALPHA = re.compile("[^A-Za-z_0-9]")


def char_level_splitter(text: str) -> list[str]:
    """
    Split the text into characters.
    This is a simple implementation that splits the text into individual characters.
    """
    return list(text)


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
    return [token.surface for token in _japanese_tagger(text)]


class GenerateDedupLSH(Filter):
    """
    Filter that uses MinHash + Locality-Sensitive Hashing (LSH) to assign
    deduplication keys to documents, allowing fast near-duplicate detection.

    Attributes:
        num_perm (int): Number of permutations (hash functions) for MinHash.
        threshold (float): Similarity threshold for tuning LSH parameters.
        tokenizer (Callable[[str], Iterable[str]]): Function to tokenize text.
        n_grams (int): n-gram size for token grouping.
        seed (int): Random seed for MinHash.
        num_bands (int): Number of LSH bands computed from threshold and num_perm.
        band_size (int): Number of hashes per band.

    Notes
    -----
    `_optimal_param` searches for the optimal number of **bands** (`b`) and
    **rows per band** (`r`) that minimise a weighted sum of false positives /
    false negatives at the specified *threshold*.
    """

    _BYTES_PER_U32: Final[int] = 4

    def __init__(
        self,
        num_perm: int = 500,
        threshold: float = 0.8,
        tokenizer: Callable[[str], Iterable[str]] = char_level_splitter,
        n_grams: int = 5,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the deduplication filter with MinHash and LSH settings.

        Args:
            num_perm: Number of hash permutations for MinHash signature length.
            threshold: Similarity threshold to decide optimal LSH parameters.
            tokenizer: Function to split text into tokens.
            n_grams: Number of tokens per n-gram for MinHash update.
            seed: Seed for hash permutation consistency.
            **kwargs: Additional keyword arguments for parent Filter.
        """
        super().__init__(**kwargs)
        self.num_perm = num_perm
        self.threshold = threshold
        self.tokenizer = tokenizer
        self.n_grams = n_grams
        self.seed = seed

        # Compute optimal number of bands and band size based on threshold
        self.num_bands, self.band_size = _optimal_param(
            threshold=self.threshold,
            num_perm=self.num_perm,
            false_negative_weight=0.5,
            false_positive_weight=0.5,
        )

    def calculate_minhash_signature(self, text: str) -> NDArray[np.uint32]:
        """
        Compute MinHash signature of input text as an array of uint32.

        Steps:
            1. Tokenize text using the provided tokenizer.
            2. Generate n-gram tokens.
            3. Update MinHash with n-gram tokens.

        Args:
            text: Input document text to be hashed.

        Returns:
            A 1D numpy array of shape (num_perm,) with dtype uint32.
        """
        tokens = self.tokenizer(text)
        n_gram_tokens = ngrams(tokens, self.n_grams)
        # Join tokens into string n-grams for hashing
        tokens = [" ".join(grams) for grams in n_gram_tokens]
        # Initialize and update RMinHash
        minhash = RMinHash(num_perm=self.num_perm, seed=self.seed)
        minhash.update(tokens)
        # Convert digest (list of ints) to numpy uint32 array
        return np.asarray(minhash.digest(), dtype=np.uint32)

    def signature_to_lsh_digest(
        self, signature: NDArray[np.uint32], band_size: int, band_idx: int
    ) -> int:
        """
        Convert a slice of the MinHash signature into an LSH digest with less memory overhead.

        This method is optimized for speed by avoiding copies:
        - We view the uint32 array as raw bytes (uint8 view).
        - We create a memoryview of the byte slice for the specified band.
        - We compute a 128-bit hash directly on the slice.

        Args:
            signature: 1D numpy array of uint32 representing MinHash signature.
            band_size: Number of hashes per LSH band.
            band_idx: Index of the band to hash (0-based).

        Returns:
            An integer representing the 128-bit hash digest of the band.

        Raises:
            AssertionError: If signature shape/dtype or band index is invalid.
        """
        assert signature.dtype == np.uint32 and signature.ndim == 1, (
            "signature must be a 1D numpy array of uint32"
        )
        assert 0 <= band_idx < self.num_bands, (
            f"band_idx {band_idx} out of range [0, {self.num_bands})"
        )
        assert len(signature) >= band_size * self.num_bands, (
            "signature length is too short for given band_size and num_bands"
        )

        # Compute byte offsets for the selected band
        start = band_idx * band_size * self._BYTES_PER_U32
        stop = start + band_size * self._BYTES_PER_U32

        # View signature as raw bytes without copy. memoryview avoids creating new bytes.
        u8 = signature.view(np.uint8)
        mv = memoryview(u8)[start:stop]  # type: ignore[arg-type]

        return xxhash.xxh128_intdigest(mv)

    def _format_lsh_key(self, band_idx: int, digest: int) -> str:
        """
        Format the LSH key for a given band index and digest.
        """
        return f"{band_idx}+{digest:032x}"

    def apply(self, document: Document) -> Document:
        """
        Decorate the document with LSH deduplication keys.

        For each band, compute the digest and format as a hex string:
            '<band_idx>+<128-bit-digest-hex>'.
        Keys are stored in document.extras['dedup_lsh'].

        Args:
            document: Document object with 'text' attribute.

        Returns:
            The same Document object with 'dedup_lsh' added in extras.
        """
        signature = self.calculate_minhash_signature(document.text)
        lsh_keys = [
            self._format_lsh_key(
                band_idx, self.signature_to_lsh_digest(signature, self.band_size, band_idx)
            )
            for band_idx in range(self.num_bands)
        ]

        document.extras["dedup_lsh"] = lsh_keys
        return document


class InlineDeduplicator(Filter):
    """
    Simple in‑memory deduplicator.

    Stores every LSH key in a local :pyclass:`set`. If any key of the incoming
    document is already present, the document is marked as duplicate via
    `document.is_rejected = True`.

    **Limitations**
    -------------
    *State is per‑process only.* Running multiple workers or machines will *not*
    share the key set – use :class:`RedisDeduplicator` or
    :class:`RedisBloomDeduplicator` for distributed setups.
    """

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


class RedisDeduplicator(Filter):
    """
    Distributed deduplicator using **plain Redis keys**.
    You have to run a Redis server and pass its connection parameters.
    """

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        key_prefix: str = "dedup",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Redis deduplicator.
        Args:
            host (str): Redis server hostname.
            port (int): Redis server port.
            db (int): Redis database number.
            key_prefix (str): Prefix for Redis keys to avoid collisions. You should use a unique prefix for each deduplication task.
            **kwargs: Additional keyword arguments for parent Filter.
        """
        super().__init__(**kwargs)
        self.rds = redis.Redis(host=host, port=port, db=db, decode_responses=False)
        self.key_prefix = key_prefix.encode()

        try:
            self.rds.ping()
        except redis.exceptions.RedisError as exc:
            raise RuntimeError(f"Cannot connect to Redis server {host}:{port}/{db}") from exc

    def apply(self, document: Document) -> Document:
        lsh_keys = document.extras.get("dedup_lsh")
        if lsh_keys is None:
            raise ValueError("Apply GenerateDedupLSH first")

        pipe = self.rds.pipeline(transaction=False)
        for k in lsh_keys:
            pipe.set(self.key_prefix + b":" + k.encode(), b"1", nx=True)
        results: list[bool | None] = pipe.execute()  # If instance already exists, it returns None

        if any(r is None for r in results):
            document.is_rejected = True
        return document


class RedisBloomDeduplicator(Filter):
    """
    Distributed deduplicator backed by **RedisBloom scalable Bloom filters**.
    You can use this filter to store-LSHs with less memory than Redis keys with the risk of false positives.

    Each *band* gets its own scalable Bloom filter on the Redis side:

    ```text
    BF.RESERVE <prefix>:<band_idx> <error> <capacity> EXPANSION <n>
    ```

    """

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        key_prefix: str = "bloomdedup",
        error_rate: float = 1e-4,
        capacity: int = 1_000_000_000,
        expansion: int = 2,
        **kwargs: Any,
    ):
        """
        Initialize the RedisBloom deduplicator.
        Args:
            host (str): Redis server hostname.
            port (int): Redis server port.
            db (int): Redis database number.
            key_prefix (str): Prefix for Redis keys to avoid collisions. You should use a unique prefix for each deduplication task.
            error_rate (float): Desired error rate for the Bloom filter.
            capacity (int): Initial capacity of the Bloom filter. Guide: set it to the expected number of unique LSH keys ~ num_doc * num_bands.
            expansion (int): Expansion factor for the Bloom filter. This is used to increase the capacity of the filter dynamically.
            **kwargs: Additional keyword arguments for parent Filter.
        """
        super().__init__(**kwargs)
        self.rds = redis.Redis(host=host, port=port, db=db)
        self.key_prefix = key_prefix.encode()

        try:
            self.rds.execute_command(
                "BF.RESERVE",
                self.key_prefix,
                error_rate,
                capacity,
                "EXPANSION",
                expansion,
            )
        except redis.ResponseError as e:
            if "exists" not in str(e):
                raise

    def apply(self, document: Document) -> Document:
        lsh_keys: list[str] | None = document.extras.get("dedup_lsh")
        if lsh_keys is None:
            raise ValueError(
                "Document does not contain LSH keys for deduplication. Please apply GenerateDedupLSH first."
            )

        key_bytes = [k.encode() for k in lsh_keys]

        # Return value of BF.MADD is [1,0,1,...] (0 = already exists, 1 = insertion successful)
        flags: Iterable[int] = self.rds.execute_command("BF.MADD", self.key_prefix, *key_bytes)
        if 0 in flags:
            document.is_rejected = True
        return document


class InlineDuplicateAnalyzer(Filter):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.docs: dict[int, Document] = dict()  # doc_id -> Document mapping
        self.hash_pool: dict[str, int] = dict()  # LSH key -> doc_id mapping

        self._current_doc_id = 0

    def apply(self, document: Document) -> Document:
        """
        Analyze duplicates inline based on the LSH keys in the document.
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
                document.extras["similar_doc"] = self.docs[self.hash_pool[lsh]].text
            else:
                self.hash_pool[lsh] = self._current_doc_id
                self.docs[self._current_doc_id] = document

        self._current_doc_id += 1
        return document
