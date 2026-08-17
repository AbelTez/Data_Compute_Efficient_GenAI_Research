"""
run_all.py
----------
Reproduces every number and every figure in the paper, in order:

  1. verify.py          pipeline self-checks (~2s, no training)
  2. run_factorial.py   the 2^3 factorial, 2 benchmarks x 8 cells x 3 seeds
  3. run_budget.py      extended-budget and equal-wall-clock follow-up
  4. make_figures.py    all five figures, from the JSON produced above

Roughly 35-45 minutes on a single CPU core. No GPU, no model hub, no
external service beyond a one-time ~1.1MB download of the public-domain
corpus used by the real-text benchmark.

Run: python3 experiments/run_all.py
"""

import hashlib
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data_cache")
CORPUS = os.path.join(DATA_DIR, "tinyshakespeare.txt")

# The URL points at a mutable branch, so the download is checked against the
# digest of the exact file these results were produced from. Without that,
# the real-text numbers silently depend on whatever upstream happens to be
# serving that day.
CORPUS_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
CORPUS_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
CORPUS_BYTES = 1_115_394


def _digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_corpus():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CORPUS):
        print(f"  downloading corpus from {CORPUS_URL}")
        try:
            with urllib.request.urlopen(CORPUS_URL, timeout=30) as r:
                data = r.read()
        except Exception as e:  # noqa: BLE001 - report any network failure plainly
            raise SystemExit(
                f"Could not fetch the corpus ({e}). Download it manually to "
                f"{CORPUS} and re-run."
            )
        with open(CORPUS, "wb") as f:
            f.write(data)
        print(f"  saved {len(data):,} bytes")

    got = _digest(CORPUS)
    size = os.path.getsize(CORPUS)
    if got != CORPUS_SHA256:
        print(f"  WARNING: corpus digest {got[:16]}... does not match the "
              f"{CORPUS_SHA256[:16]}... these results were produced from.")
        print(f"  WARNING: real-text numbers will not match the paper exactly.")
    else:
        print(f"  corpus verified: {size:,} bytes, sha256 {got[:16]}...")


def main():
    t0 = time.time()
    steps = [
        ("Verifying the pipeline", "verify"),
        ("Fetching the real corpus", None),
        ("Main study: 2^3 factorial", "run_factorial"),
        ("Follow-up: budget study", "run_budget"),
        ("Rendering figures", "make_figures"),
    ]
    for i, (title, module) in enumerate(steps, 1):
        print(f"\n{'#' * 72}\n# [{i}/{len(steps)}] {title}\n{'#' * 72}")
        if module is None:
            fetch_corpus()
            continue
        mod = __import__(f"experiments.{module}", fromlist=["main"])
        mod.main()

    mins = (time.time() - t0) / 60
    print(f"\n{'#' * 72}")
    print(f"# Done in {mins:.1f} minutes.")
    print(f"# results/factorial.json, results/budget.json, figures/*.png")
    print(f"# Regenerate the paper with: python3 experiments/make_paper.py")
    print(f"{'#' * 72}")


if __name__ == "__main__":
    main()
