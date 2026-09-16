import json
import re
from pathlib import Path

import numpy as np
import pytest

from hojichar.core.models import Document
from hojichar.filters import deduplication as module

# Golden values from datasketch 1.6.5; dependency versions are recorded in the fixture.
# Do not regenerate these automatically when upgrading dependencies: changes affect
# compatibility with persisted LSH keys and must be reviewed.
_LSH_PARAMS_BASELINE = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "deduplication_lsh_params.json").read_text()
)

# Full output snapshots cover tokenization, MinHash, band slicing, and key formatting.
# Review compatibility before changing these expected keys; never regenerate in tests.
_LSH_KEYS_BASELINE = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "deduplication_lsh_keys_v2.json").read_text(
        encoding="utf-8"
    )
)


@pytest.mark.parametrize("case", _LSH_KEYS_BASELINE["cases"], ids=lambda case: case["id"])
def test_natural_language_lsh_keys_match_baseline(case):
    settings = dict(case["settings"])
    if "tokenizer" in settings:
        settings["tokenizer"] = getattr(module, settings["tokenizer"])
    filt = module.GenerateDedupLSH(**settings)
    document = filt.apply(Document(text=case["text"]))
    assert document.extras["dedup_lsh"] == case["expected_keys"]


@pytest.mark.parametrize("num_perm,threshold,num_bands,band_size", _LSH_PARAMS_BASELINE["cases"])
def test_automatic_lsh_params_match_baseline(num_perm, threshold, num_bands, band_size):
    filt = module.GenerateDedupLSH(num_perm=num_perm, threshold=threshold)
    assert (filt.num_bands, filt.band_size) == (num_bands, band_size)


def test_char_level_splitter():
    assert module.char_level_splitter("abc") == ["a", "b", "c"]
    assert module.char_level_splitter("") == []


def test_non_alpha_num_splitter():
    s = "Hello, 123-world!!"
    assert module.non_alpha_num_splitter(s) == ["Hello", "123", "world"]


def test_ngrams():
    assert list(module._ngrams(iter("abcde"), 3)) == [
        ("a", "b", "c"),
        ("b", "c", "d"),
        ("c", "d", "e"),
    ]
    assert list(module._ngrams(iter("ab"), 3)) == []


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


@pytest.mark.parametrize(
    "text,n_grams,expected_tokens",
    [
        ("", 3, []),
        ("ab", 3, []),
        ("abc", 3, ["abc"]),
        ("a b\nc", 3, ["a b", " b\n", "b\nc"]),
        ("日本語😀", 2, ["日本", "本語", "語😀"]),
        ("a😀", 1, ["a", "😀"]),
    ],
)
def test_character_ngrams_use_text_slices(text, n_grams, expected_tokens):
    filt = module.GenerateDedupLSH(num_bands=3, band_size=4, n_grams=n_grams)
    reference = module.RMinHash(num_perm=12, seed=42)
    reference.update(expected_tokens)
    np.testing.assert_array_equal(filt.calculate_minhash_signature(text), reference.digest())


def test_custom_tokenizer_keeps_space_joined_ngrams():
    calls = []

    def tokenize(text):
        calls.append(text)
        return iter(["ab", "c", "日本語", "😀"])

    filt = module.GenerateDedupLSH(num_bands=3, band_size=4, n_grams=2, tokenizer=tokenize)
    reference = module.RMinHash(num_perm=12, seed=42)
    reference.update(["ab c", "c 日本語", "日本語 😀"])
    np.testing.assert_array_equal(filt.calculate_minhash_signature("input"), reference.digest())
    assert calls == ["input"]


@pytest.mark.parametrize("n_grams", [0, -1])
def test_nonpositive_ngram_size(n_grams):
    with pytest.raises(ValueError, match="n_grams must be positive"):
        module.GenerateDedupLSH(n_grams=n_grams)


@pytest.mark.parametrize("text", ["", "ab", "日本語とEnglish 😀\nテスト"])
@pytest.mark.parametrize("settings", [{}, {"num_bands": 3, "band_size": 4}])
def test_fast_keys_match_public_signature_api(text, settings):
    filt = module.GenerateDedupLSH(**settings)
    signature = filt.calculate_minhash_signature(text)
    expected = [
        filt._format_lsh_key(i, filt.signature_to_lsh_digest(signature, filt.band_size, i))
        for i in range(filt.num_bands)
    ]
    assert filt.apply(Document(text=text)).extras["dedup_lsh"] == expected


@pytest.mark.parametrize("num_perm,threshold", [(1, -1.0), (500, 0.8), (1000, 2.0)])
def test_explicit_bands_ignore_automatic_settings(monkeypatch, num_perm, threshold):
    def fail_if_called(**kwargs):
        pytest.fail("Explicit bands must bypass automatic parameter selection")

    monkeypatch.setattr(module, "_optimal_param", fail_if_called)
    filt = module.GenerateDedupLSH(
        num_perm=num_perm,
        threshold=threshold,
        num_bands=3,
        band_size=4,
    )
    assert filt.num_perm == 12
    assert (filt.num_bands, filt.band_size) == (3, 4)
    assert filt.calculate_minhash_signature("hello world").shape == (12,)
    keys = filt.apply(Document(text="hello world")).extras["dedup_lsh"]
    assert len(keys) == 3
    reference = module.GenerateDedupLSH(num_bands=3, band_size=4)
    assert keys == reference.apply(Document(text="hello world")).extras["dedup_lsh"]


@pytest.mark.parametrize("kwargs", [{"num_bands": 3}, {"band_size": 4}])
def test_incomplete_explicit_bands(kwargs):
    with pytest.raises(ValueError, match="must be provided together"):
        module.GenerateDedupLSH(**kwargs)


@pytest.mark.parametrize(
    "num_bands,band_size",
    [(0, 2), (2, 0), (-1, 2), (2, -1)],
)
def test_invalid_explicit_bands(num_bands, band_size):
    with pytest.raises(ValueError, match="must be positive"):
        module.GenerateDedupLSH(num_bands=num_bands, band_size=band_size)


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
    assert key == "v2:2+000000000000000000000000001a2b3c"
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
    pattern = re.compile(r"^v2:\d+\+[0-9a-f]{32}$")
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
