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
  - Treat the `r` integers as raw bytes, hash them with xxhash‑128, and format as v2:<band_idx>+<digest32hex>.
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
import struct
import sys
from collections import deque
from itertools import islice
from typing import Any, Callable, Final, Iterable, Optional, cast

import numpy as np
from numpy.typing import NDArray

try:
    import redis
    import xxhash
    from datasketch.lsh import _optimal_param  # type: ignore
    from rensa import RMinHash  # type: ignore

    is_loaded_dedup = True
except ImportError:
    is_loaded_dedup = False


from hojichar import Document, Filter

_japanese_tagger: Optional["fugashi.Tagger"] = None  # type: ignore[name-defined] # noqa: F821
NON_ALPHA = re.compile("[^A-Za-z_0-9]")
IS_LOADED_DEDUP_ERROR_MSG = (
    "Failed to import redis, xxhash, rensa, or datasketch. "
    "Please install the extra dependencies with `pip install 'hojichar[dedup]'`"
)


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


def _ngrams(tokens: Iterable[str], n: int) -> Iterable[tuple[str, ...]]:
    """Yield sliding windows of *n* tokens."""
    iterator = iter(tokens)
    window = deque(islice(iterator, n), maxlen=n)
    if len(window) == n:
        yield tuple(window)
    for token in iterator:
        window.append(token)
        yield tuple(window)


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
        num_bands (int): Number of LSH bands, specified explicitly or computed automatically.
        band_size (int): Number of hashes per band.

    Notes
    -----
    When band parameters are omitted, `_optimal_param` searches for the optimal
    number of **bands** (`b`) and
    **rows per band** (`r`) that minimise a weighted sum of false positives /
    false negatives at the specified *threshold*.

    HojiChar 0.18.0 introduces ``v2:`` keys using Rensa 0.5, which was roughly
    twice as fast for long documents in our benchmarks. Hashes differ from
    HojiChar 0.17.x. Please rebuild your deduplication fingerprints,
    or keep using HojiChar 0.17.x if you want to use existing LSH pool.
    """

    _BYTES_PER_U32: Final[int] = 4
    _LSH_KEY_VERSION: Final[str] = "v2"

    def __init__(
        self,
        num_perm: int = 500,
        threshold: float = 0.8,
        tokenizer: Callable[[str], Iterable[str]] = char_level_splitter,
        n_grams: int = 5,
        seed: int = 42,
        *,
        num_bands: Optional[int] = None,
        band_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the deduplication filter with MinHash and LSH settings.

        Args:
            num_perm: Number of hash permutations for MinHash signature length.
                Ignored when num_bands and band_size are both provided.
            threshold: Similarity threshold to decide optimal LSH parameters.
                Ignored when num_bands and band_size are both provided.
            tokenizer: Function to split text into tokens.
            n_grams: Number of tokens per n-gram for MinHash update.
            seed: Seed for hash permutation consistency.
            num_bands: Explicit number of LSH bands. Must be a positive integer
                and provided together with band_size.
            band_size: Explicit number of hashes per band. Must be a positive integer
                and provided together with num_bands. When both are provided,
                automatic selection is bypassed and the MinHash signature length
                (self.num_perm) is set to num_bands * band_size.
            **kwargs: Additional keyword arguments for parent Filter.
        """
        super().__init__(**kwargs)
        if not is_loaded_dedup:
            raise ImportError(IS_LOADED_DEDUP_ERROR_MSG)
        if n_grams <= 0:
            raise ValueError("n_grams must be positive")
        self.num_perm = num_perm
        self.threshold = threshold
        self.tokenizer = tokenizer
        self.n_grams = n_grams
        self.seed = seed

        if num_bands is None and band_size is None:
            self.num_bands, self.band_size = _optimal_param(
                threshold=self.threshold,
                num_perm=self.num_perm,
                false_negative_weight=0.5,
                false_positive_weight=0.5,
            )
        else:
            if num_bands is None or band_size is None:
                raise ValueError("num_bands and band_size must be provided together")
            if num_bands <= 0 or band_size <= 0:
                raise ValueError("num_bands and band_size must be positive")
            self.num_bands = num_bands
            self.band_size = band_size
            self.num_perm = num_bands * band_size

    def _calculate_minhash_digest(self, text: str) -> list[int]:
        """Compute the raw digest shared by the array API and the LSH fast path."""
        if self.tokenizer is char_level_splitter:
            n = self.n_grams
            tokens = [text[i : i + n] for i in range(len(text) - n + 1)]
        else:
            tokens = [" ".join(grams) for grams in _ngrams(self.tokenizer(text), self.n_grams)]
        minhash = RMinHash(num_perm=self.num_perm, seed=self.seed)
        minhash.update(tokens)
        return cast(list[int], minhash.digest())

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
        return np.asarray(self._calculate_minhash_digest(text), dtype=np.uint32)

    def _sig_bytes_le(self, sig: NDArray[np.uint32]) -> memoryview:
        """
        Return the signature as *little-endian* byte view.
        Platform‑independent way to get bytes from a numpy array.
        """
        if sys.byteorder == "little":
            # amd64 / arm64 (little)
            return memoryview(cast(Any, sig)).cast("B")  # zero-copy
        else:
            # big-endian CPU
            return memoryview(cast(Any, sig.byteswap())).cast("B")  # 1 copy

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
        mv = self._sig_bytes_le(signature)[start:stop]  # slice view

        return xxhash.xxh128_intdigest(mv)

    def _format_lsh_key(self, band_idx: int, digest: int) -> str:
        """
        Format the LSH key with the HojiChar scheme version, band index, and digest.
        """
        return f"{self._LSH_KEY_VERSION}:{band_idx}+{digest:032x}"

    def apply(self, document: Document) -> Document:
        """
        Decorate the document with LSH deduplication keys.

        For each band, compute the digest and format as a hex string:
            'v2:<band_idx>+<128-bit-digest-hex>'.
        Keys are stored in document.extras['dedup_lsh'].

        Args:
            document: Document object with 'text' attribute.

        Returns:
            The same Document object with 'dedup_lsh' added in extras.
        """
        digest = self._calculate_minhash_digest(document.text)
        # Pack once in little-endian order without an intermediate NumPy array.
        signature_bytes = memoryview(struct.pack(f"<{self.num_perm}I", *digest))
        band_bytes = self.band_size * self._BYTES_PER_U32
        lsh_keys = [
            f"{self._LSH_KEY_VERSION}:{band_idx}+"
            + xxhash.xxh128_hexdigest(
                signature_bytes[band_idx * band_bytes : (band_idx + 1) * band_bytes]
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
        if not is_loaded_dedup:
            raise ImportError(IS_LOADED_DEDUP_ERROR_MSG)
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
        expected_docs: int,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        key_prefix: str = "bloomdedup",
        error_rate: float = 1e-7,
        expansion: int = 2,
        num_bands: int | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the RedisBloom deduplicator.
        Args:
            expected_docs (int): Expected number of documents to deduplicate. This is used to set the initial capacity of the Bloom filter.
            host (str): Redis server hostname.
            port (int): Redis server port.
            db (int): Redis database number.
            key_prefix (str): Prefix for Redis keys to avoid collisions. You should use a unique prefix for each deduplication task.
            error_rate (float): Desired error rate for the Bloom filter.
            expansion (int): Expansion factor for the Bloom filter. This is used to increase the capacity of the filter dynamically.
            num_bands (int | None): Number of bands to use for LSH to calculate the capacity of BloomFilter. If None, it will be set to 32.
            **kwargs: Additional keyword arguments for parent Filter.
        """
        if not is_loaded_dedup:
            raise ImportError(IS_LOADED_DEDUP_ERROR_MSG)
        super().__init__(**kwargs)
        self.rds = redis.Redis(host=host, port=port, db=db)
        self.key_prefix = key_prefix.encode()

        _num_bands = num_bands if num_bands is not None else 32

        try:
            self.rds.execute_command(
                "BF.RESERVE",
                self.key_prefix,
                error_rate,
                expected_docs * _num_bands,
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
