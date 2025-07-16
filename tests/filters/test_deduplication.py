import re

import numpy as np
import pytest

from hojichar.core.models import Document
from hojichar.filters import deduplication as module


def test_char_level_splitter():
    assert module.char_level_splitter("abc") == ["a", "b", "c"]
    assert module.char_level_splitter("") == []


def test_non_alpha_num_splitter():
    s = "Hello, 123-world!!"
    assert module.non_alpha_num_splitter(s) == ["Hello", "123", "world"]


def test_japanese_word_splitter_roundtrip():
    fugashi = pytest.importorskip("fugashi")  # noqa
    text = "これはテスト文章です"
    tokens = module.japanese_word_splitter(text)
    assert isinstance(tokens, list)
    assert "".join(tokens) == text


def test_calculate_minhash_signature_length_and_dtype():
    filt = module.GenerateDedupLSH(num_perm=10, threshold=0.5)
    sig = filt.calculate_minhash_signature("abcdefghij")
    assert isinstance(sig, np.ndarray)
    assert sig.shape == (10,)
    assert sig.dtype == np.uint32


def test_signature_to_lsh_digest_repeatable():
    filt = module.GenerateDedupLSH(num_perm=10, threshold=0.5)
    sig = np.arange(10, dtype=np.uint32)
    digest1 = filt.signature_to_lsh_digest(sig, filt.band_size, 0)
    digest2 = filt.signature_to_lsh_digest(sig, filt.band_size, 0)
    assert isinstance(digest1, int)
    assert digest1 == digest2


def test_format_lsh_key_zero_padding():
    filt = module.GenerateDedupLSH(num_perm=10, threshold=0.5)
    key = filt._format_lsh_key(2, 0x1A2B3C)
    assert key.startswith("2+")
    hexpart = key.split("+", 1)[1]
    assert len(hexpart) == 32
    assert hexpart.endswith("00000000001a2b3c")


def test_apply_adds_lsh_keys_to_document():
    filt = module.GenerateDedupLSH(num_perm=10, threshold=0.5)
    doc = Document(text="hello world")
    out = filt.apply(doc)
    assert out is doc
    keys = doc.extras.get("dedup_lsh")
    assert isinstance(keys, list)
    assert len(keys) == filt.num_bands
    pattern = re.compile(r"^\d+\+[0-9a-f]{32}$")
    for k in keys:
        assert pattern.match(k), f"invalid LSH key format: {k}"


def test_inline_deduplicator_marks_exact_duplicate():
    gen = module.GenerateDedupLSH(num_perm=10, threshold=0.5)
    dedup = module.InlineDeduplicator()

    doc1 = Document(text="duplicate text")
    doc2 = Document(text="duplicate text")

    gen.apply(doc1)
    dedup.apply(doc1)
    assert not getattr(doc1, "is_rejected", False)

    gen.apply(doc2)
    dedup.apply(doc2)
    assert getattr(doc2, "is_rejected", False)


def test_near_duplicate_detection():
    # ある程度似ている文で重複検知されるか
    t1 = "The quick brown fox jumps over the lazy dog"
    t2 = "The quick brown fox jumps over the lazy dog."
    # パラメータを調整して検知しやすくする
    gen = module.GenerateDedupLSH(
        num_perm=50,
        threshold=0.5,
        tokenizer=module.char_level_splitter,
        n_grams=3,
        seed=0,
    )
    dedup = module.InlineDeduplicator()

    doc1 = Document(text=t1)
    doc2 = Document(text=t2)
    gen.apply(doc1)
    dedup.apply(doc1)
    gen.apply(doc2)
    dedup.apply(doc2)

    assert getattr(doc2, "is_rejected", False), "Near-duplicate should be rejected"


def test_non_duplicate_not_marked():
    # 類似度が低い文ではリジェクトされないこと
    t1 = "Completely different text with zero overlap"
    t2 = "Nothing in common here at all"
    gen = module.GenerateDedupLSH(
        num_perm=50,
        threshold=0.8,
        tokenizer=module.char_level_splitter,
        n_grams=3,
        seed=0,
    )
    dedup = module.InlineDeduplicator()

    doc1 = Document(text=t1)
    doc2 = Document(text=t2)
    gen.apply(doc1)
    dedup.apply(doc1)
    gen.apply(doc2)
    dedup.apply(doc2)

    assert not getattr(doc2, "is_rejected", False), "Dissimilar text should not be rejected"
