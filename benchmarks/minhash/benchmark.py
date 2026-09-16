"""Reproduce the fixed-input MinHash/LSH benchmark with isolated Rensa versions.

Run with Python 3.11 and the versions in environment.json:
  python benchmark.py --repo /path/to/HojiChar --rensa05 /path/to/rensa-0.5-target
The base interpreter must contain rensa 0.2.7. Install rensa 0.5.0 with
`uv pip install --target /path/to/rensa-0.5-target --no-deps rensa==0.5.0`.
"""

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import random
import statistics
import struct
import subprocess
import sys
import time
import types
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METHODS = ["hojichar_018", "hojichar_0173", "datasketch", "datatrove"]


def load_module(name):
    metadata = json.loads((ROOT / "source_metadata.json").read_text())[name]
    source = subprocess.check_output(
        [
            "git",
            "-C",
            os.environ["HOJICHAR_BENCH_REPO"],
            "show",
            metadata["git_commit"] + ":hojichar/filters/deduplication.py",
        ]
    )
    assert hashlib.sha256(source).hexdigest() == metadata["sha256"]
    module = types.ModuleType(name)
    exec(compile(source, name + ".py", "exec"), module.__dict__)
    return module


def make_function(method):
    import xxhash

    from hojichar import Document

    if method.startswith("hojichar"):
        module = load_module(method)
        assert version("rensa") == ("0.5.0" if method == "hojichar_018" else "0.2.7")
        if method == "hojichar_018":
            filt = module.GenerateDedupLSH(num_bands=25, band_size=20)
        else:
            filt = module.GenerateDedupLSH(num_perm=500, threshold=0.8)
            # Fixed band geometry, set outside timed processing; original implementation unchanged.
            filt.num_bands, filt.band_size = 25, 20
        return lambda text: filt.apply(Document(text=text)).extras["dedup_lsh"]

    def encode_bands(bands):
        # Common 32-bit band-to-string adapter for the two comparison libraries.
        return [
            f"{i}+{xxhash.xxh128_hexdigest(struct.pack('<20I', *band))}"
            for i, band in enumerate(bands)
        ]

    if method == "datasketch":
        from datasketch import MinHash

        # Use the vectorized API and reusable coefficients, avoiding a slow per-token baseline.
        permutations = MinHash(num_perm=500, seed=42).permutations

        def run(text):
            mh = MinHash(
                num_perm=500, seed=42, permutations=permutations, hashfunc=xxhash.xxh32_intdigest
            )
            mh.update_batch([text[i : i + 5].encode("utf-8") for i in range(len(text) - 4)])
            return encode_bands(mh.hashvalues.reshape(25, 20))

        return run

    from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupSignature
    from datatrove.utils.hashing import HashConfig
    from datatrove.utils.text import TextNormConfig
    from datatrove.utils.word_tokenizers import WordTokenizer

    class CharacterTokenizer(WordTokenizer):
        def word_tokenize(self, text):
            return list(text)

        def sent_tokenize(self, text):
            return [text]

        def span_tokenize(self, text):
            return [(0, len(text))]

    norm = TextNormConfig(
        lowercase=False,
        norm_whitespace=False,
        remove_punctuation=False,
        norm_unicode_diacritics=False,
        norm_numbers=False,
        norm_weekdays=False,
        norm_monthnames=False,
    )
    config = MinhashConfig(
        n_grams=5,
        num_buckets=25,
        hashes_per_bucket=20,
        seed=42,
        norm_config=norm,
        hash_config=HashConfig(precision=32, hash_fc="xxhash"),
    )
    filt = MinhashDedupSignature(
        output_folder=str(ROOT / "unused-datatrove-output"),
        config=config,
        language=CharacterTokenizer(),
    )
    _ = filt.parameters

    def run(text):
        shingles = filt.get_shingles(text)
        return encode_bands(filt.get_signature(shingles))

    return run


def worker(method):
    fn = make_function(method)
    cases = {x["id"]: x for x in json.loads((ROOT / "inputs.json").read_text())}
    loops = {}
    print(json.dumps({"ready": method}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        case = cases[request["case"]]
        docs = [d["text"] for d in case["documents"]]

        def block(count):
            start = time.perf_counter_ns()
            for _ in range(count):
                for text in docs:
                    fn(text)
            return (time.perf_counter_ns() - start) / 1e9

        if request["case"] not in loops:
            # Check deterministic output and expected band count before calibration.
            first = fn(docs[0])
            assert len(first) == 25 and all(isinstance(k, str) for k in first)
            assert first == fn(docs[0])
            count = 1
            while block(count) < 0.08:
                count *= 2
            loops[request["case"]] = count
        gc.collect()
        gc.disable()
        try:
            count = loops[request["case"]]
            elapsed = block(count)
        finally:
            gc.enable()
        print(
            json.dumps(
                {
                    "seconds_per_document": elapsed / count / len(docs),
                    "loops": count,
                    "docs_per_loop": len(docs),
                }
            ),
            flush=True,
        )


def main(args):
    cases = json.loads((ROOT / "inputs.json").read_text())
    processes = {}
    try:
        for method in METHODS:
            env = dict(os.environ)
            env["HOJICHAR_BENCH_REPO"] = args.repo
            env["PYTHONPATH"] = (
                args.rensa05 + os.pathsep if method == "hojichar_018" else ""
            ) + args.repo
            env["OMP_NUM_THREADS"] = env["OPENBLAS_NUM_THREADS"] = env["MKL_NUM_THREADS"] = "1"
            p = subprocess.Popen(
                [sys.executable, __file__, "--worker", method],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                env=env,
            )
            ready = p.stdout.readline()
            if not ready:
                raise RuntimeError(f"{method} failed to start")
            assert json.loads(ready)["ready"] == method
            processes[method] = p
        samples = []
        rng = random.Random(42)
        for round_id in range(5):
            tasks = [(case, method) for case in cases for method in METHODS]
            rng.shuffle(tasks)
            for case, method in tasks:
                p = processes[method]
                p.stdin.write(json.dumps({"case": case["id"]}) + "\n")
                p.stdin.flush()
                line = p.stdout.readline()
                if not line:
                    raise RuntimeError(f"{method} exited during {case['id']}")
                sample = json.loads(line)
                sample.update(
                    method=method,
                    language=case["language"],
                    characters=case["characters"],
                    round=round_id,
                )
                samples.append(sample)
            (ROOT / "samples.json").write_text(json.dumps(samples, indent=2))
            print(f"Completed round {round_id + 1}/5 ({len(samples)} measurements)", flush=True)
        results = []
        for case in cases:
            for method in METHODS:
                values = [
                    x["seconds_per_document"]
                    for x in samples
                    if x["method"] == method
                    and x["language"] == case["language"]
                    and x["characters"] == case["characters"]
                ]
                results.append(
                    {
                        "method": method,
                        "language": case["language"],
                        "characters": case["characters"],
                        "median_ms": statistics.median(values) * 1e3,
                        "min_ms": min(values) * 1e3,
                        "max_ms": max(values) * 1e3,
                        "documents_per_second": 1 / statistics.median(values),
                    }
                )
        (ROOT / "throughput_data.json").write_text(json.dumps(results, indent=2))
        with (ROOT / "throughput_data.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        metadata = {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": {
                p: version(p)
                for p in ["numpy", "datasketch", "datatrove", "xxhash", "mmh3", "matplotlib"]
            },
            "rensa": {"hojichar_018": "0.5.0", "hojichar_0173": "0.2.7"},
            "num_hashes": 500,
            "bands": 25,
            "rows_per_band": 20,
            "n_grams": 5,
            "seed": 42,
            "rounds": 5,
            "documents_per_length": 3,
            "calibration_seconds_per_block": 0.08,
            "scope": "text to LSH band keys; includes tokenization, n-grams, hashing, MinHash, encoding; excludes initialization, file/Redis I/O, search and duplicate decisions",
            "datasketch_adapter": "update_batch, cached permutations, xxhash32; direct character slices",
            "datatrove_adapter": "original get_shingles/get_signature; character tokenizer; disabled normalization; 32-bit xxhash; band compression adapter",
            "execution": "sequential measurements; seeded shuffled order across four persistent isolated workers; GC disabled during timing; one numeric thread",
        }
        (ROOT / "environment.json").write_text(json.dumps(metadata, indent=2))
    finally:
        for p in processes.values():
            p.stdin.close()
            p.wait(timeout=30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=METHODS)
    parser.add_argument("--repo")
    parser.add_argument("--rensa05")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    else:
        main(args)
