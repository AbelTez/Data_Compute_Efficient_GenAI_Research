"""
run_factorial.py
----------------
The main study: a full 2^3 factorial over the three components of CG-PED --
curriculum ordering (C), LoRA (L), and knowledge distillation (D) -- run at a
fixed step budget on two benchmarks, three seeds per cell.

Why a factorial rather than a chain of pairwise ablations: the research
question is whether the three techniques COMPOSE. That question is about the
interaction between factors, and an interaction cannot be read off a set of
one-factor-at-a-time comparisons. With all eight cells you get each factor's
main effect, and you can compare what the composition actually achieves
against what the individual effects predict it should achieve. The gap between
those two numbers is the paper's headline.

Cells are named by which factors are ON, e.g. "---" = naive, "CLD" = full
CG-PED. Everything else is held identical: same architecture, same optimizer,
same learning rate, same batch size, same step budget, same seed per row.

Run: python3 experiments/run_factorial.py
"""

import itertools
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import (SyntheticCompositionalDataset, SyntheticConfig,
                      RealTextDataset, RealTextConfig)
from src.trainer import RunConfig, run_training, train_teacher
from src.train_utils import Timer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
DATA_DIR = os.path.join(ROOT, "data_cache")
CORPUS = os.path.join(DATA_DIR, "tinyshakespeare.txt")

SEEDS = (7, 17, 27)
N_STEPS = 300
EVAL_EVERY = 15

# The eight cells of the 2^3 design.
FACTORS = ("C", "L", "D")
CELLS = [
    "".join(f if on else "-" for f, on in zip(FACTORS, combo))
    for combo in itertools.product([False, True], repeat=3)
]

CELL_LABELS = {
    "---": "naive (random order, full fine-tune, no KD)",
    "C--": "curriculum only",
    "-L-": "LoRA only",
    "--D": "distillation only",
    "CL-": "curriculum + LoRA",
    "C-D": "curriculum + distillation",
    "-LD": "LoRA + distillation",
    "CLD": "full CG-PED",
}


def cell_flags(cell):
    return {
        "use_curriculum": "C" in cell,
        "use_lora": "L" in cell,
        "use_distillation": "D" in cell,
    }


def make_config(cell, seed, benchmark, max_len):
    return RunConfig(
        name=f"{benchmark}:{cell}:seed{seed}",
        seed=seed,
        n_steps=N_STEPS,
        eval_every=EVAL_EVERY,
        lr=4e-3,
        batch_size=32,
        max_len=max_len,
        lora_r=4,
        lora_alpha=8,
        kd_temperature=2.0,
        kd_alpha=0.5,
        teacher_steps=250,
        thresholds=(3.6, 3.55, 3.5) if benchmark == "synthetic" else (2.6, 2.4, 2.2),
        **cell_flags(cell),
    )


def build_dataset(benchmark):
    """Data generation is seeded independently of training so that every cell
    in a given row sees byte-identical data."""
    if benchmark == "synthetic":
        return SyntheticCompositionalDataset(SyntheticConfig(seed=13))
    return RealTextDataset(RealTextConfig(corpus_path=CORPUS, chunk_len=64,
                                          max_examples=4000, seed=13))


def summarise(values):
    return {
        "mean": round(statistics.mean(values), 4),
        "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "n": len(values),
    }


def run_benchmark(benchmark, device="cpu"):
    print(f"\n{'=' * 72}\n{benchmark.upper()} benchmark — 8 cells x {len(SEEDS)} seeds\n{'=' * 72}")
    dataset = build_dataset(benchmark)
    print(f"  {len(dataset.train)} train / {len(dataset.val)} val examples, "
          f"vocab {dataset.cfg.vocab_size}, max_len {dataset.cfg.max_len}")

    # One teacher per seed, shared by the four distillation cells in that row.
    teachers, teacher_meta = {}, {}
    for seed in SEEDS:
        cfg = make_config("--D", seed, benchmark, dataset.cfg.max_len)
        t = Timer()
        with t:
            teacher, info = train_teacher(dataset, cfg, device=device)
        teachers[seed] = teacher
        teacher_meta[seed] = {**info, "wall_clock_sec": round(t.elapsed, 2),
                              "params": sum(p.numel() for p in teacher.parameters())}
        print(f"  teacher seed={seed}: val_loss={info['val_loss']:.4f} "
              f"({t.elapsed:.1f}s, {teacher_meta[seed]['params']:,} params)")

    cells = {}
    for cell in CELLS:
        runs = []
        for seed in SEEDS:
            cfg = make_config(cell, seed, benchmark, dataset.cfg.max_len)
            teacher = teachers[seed] if cfg.use_distillation else None
            res = run_training(dataset, cfg, teacher=teacher, device=device)
            runs.append({
                "seed": seed,
                "final_val_loss": res["final_val_loss"],
                "final_val_ppl": res["final_val_ppl"],
                "wall_clock_sec": res["wall_clock_sec"],
                "trainable_params": res["trainable_params"],
                "history": res["history"],
            })
        losses = [r["final_val_loss"] for r in runs]
        cells[cell] = {
            "label": CELL_LABELS[cell],
            "flags": cell_flags(cell),
            "trainable_params": runs[0]["trainable_params"],
            "total_params": None,
            "val_loss": summarise(losses),
            "val_ppl": summarise([r["final_val_ppl"] for r in runs]),
            "wall_clock_sec": summarise([r["wall_clock_sec"] for r in runs]),
            "runs": runs,
        }
        s = cells[cell]["val_loss"]
        print(f"  {cell}  {CELL_LABELS[cell]:<44} "
              f"loss {s['mean']:.4f} +/- {s['std']:.4f}   "
              f"[{s['min']:.4f}, {s['max']:.4f}]   "
              f"params {cells[cell]['trainable_params']:,}")

    return {
        "benchmark": benchmark,
        "seeds": list(SEEDS),
        "n_steps": N_STEPS,
        "n_train": len(dataset.train),
        "n_val": len(dataset.val),
        "vocab_size": dataset.cfg.vocab_size,
        "teacher": teacher_meta,
        "cells": cells,
        "analysis": analyse(cells),
    }


def analyse(cells):
    """Main effects, and the composition gap that is the paper's headline.

    A factor's main effect is the mean change in validation loss across the
    four pairs of cells that differ only in that factor. Negative is better.

    The additive prediction for the full composition is the naive cell plus
    the three main effects. Comparing it to the cell actually observed says
    whether composing the three techniques delivers what they deliver
    separately (gap ~ 0), more (gap < 0, synergy), or less (gap > 0).
    """
    m = {c: cells[c]["val_loss"]["mean"] for c in cells}

    effects = {}
    for i, f in enumerate(FACTORS):
        pairs = []
        for cell in cells:
            if cell[i] == "-":
                on = cell[:i] + f + cell[i + 1:]
                pairs.append(m[on] - m[cell])
        effects[f] = {
            "main_effect": round(statistics.mean(pairs), 4),
            "per_pair": [round(p, 4) for p in pairs],
        }

    naive, full = m["---"], m["CLD"]
    additive = naive + sum(effects[f]["main_effect"] for f in FACTORS)
    pooled_std = statistics.mean([cells[c]["val_loss"]["std"] for c in cells])

    return {
        "main_effects": effects,
        "naive_mean": round(naive, 4),
        "full_cgped_mean": round(full, 4),
        "additive_prediction": round(additive, 4),
        "composition_gap": round(full - additive, 4),
        "observed_vs_naive": round(full - naive, 4),
        "pooled_within_cell_std": round(pooled_std, 4),
        "param_reduction_pct": round(
            100 * (cells["---"]["trainable_params"] - cells["CLD"]["trainable_params"])
            / cells["---"]["trainable_params"], 1),
    }


def main():
    torch.set_num_threads(1)  # keep CPU reduction order stable across machines
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(CORPUS):
        raise SystemExit(f"Missing corpus at {CORPUS}. Run experiments/run_all.py, "
                         f"which fetches it once.")

    t0 = time.time()
    out = {}
    for benchmark in ("synthetic", "real"):
        out[benchmark] = run_benchmark(benchmark)

    out["meta"] = {
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "total_wall_clock_sec": round(time.time() - t0, 1),
        "seeds": list(SEEDS),
        "n_steps": N_STEPS,
    }

    path = os.path.join(RESULTS_DIR, "factorial.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'=' * 72}\nHEADLINE\n{'=' * 72}")
    for benchmark in ("synthetic", "real"):
        a = out[benchmark]["analysis"]
        print(f"\n{benchmark}:")
        print(f"  naive                     {a['naive_mean']:.4f}")
        print(f"  full CG-PED               {a['full_cgped_mean']:.4f}  "
              f"({a['observed_vs_naive']:+.4f} vs naive)")
        print(f"  additive prediction       {a['additive_prediction']:.4f}")
        print(f"  composition gap           {a['composition_gap']:+.4f}  "
              f"(within-cell sd {a['pooled_within_cell_std']:.4f})")
        print(f"  trainable params saved    {a['param_reduction_pct']}%")
        for f_, e in a["main_effects"].items():
            print(f"  main effect {f_}             {e['main_effect']:+.4f}")

    print(f"\nWrote {path}  ({out['meta']['total_wall_clock_sec']}s total)")


if __name__ == "__main__":
    main()
