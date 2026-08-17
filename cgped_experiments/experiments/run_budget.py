"""
run_budget.py
-------------
The follow-up to the factorial: if full CG-PED is behind the naive baseline at
an equal STEP budget, does it catch up when given more steps? And what happens
under the budget that actually constrains a researcher without a GPU -- equal
WALL CLOCK rather than equal steps?

Both questions are asked because "equal steps" flatters CG-PED: its
distillation arm runs a forward pass through a teacher roughly seven times the
student's size at every step, so one CG-PED step costs about twice one naive
step. Reporting only the equal-step comparison would be reporting the budget
that makes the method look best.

Also produces the qualitative samples for the real-corpus section.

Run: python3 experiments/run_budget.py   (needs results/factorial.json)
"""

import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trainer import run_training, train_teacher
from experiments.run_factorial import (SEEDS, N_STEPS, make_config,
                                       build_dataset, summarise)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")

EXTENDED_STEPS = 900
PROMPT = "ROMEO:\n"


@torch.no_grad()
def generate_sample(model, dataset, prompt, n_chars=140, temperature=0.8, device="cpu"):
    model.eval()
    ids = [1] + [dataset.stoi.get(ch, 1) for ch in prompt]
    ids = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(n_chars):
        if ids.shape[1] >= dataset.cfg.max_len:
            break
        logits = model(ids)
        probs = torch.softmax(logits[0, -1] / temperature, dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
    model.train()
    return dataset.decode(ids[0].tolist())


def first_step_at_or_below(history, target):
    for h in history:
        if h["val_loss"] <= target:
            return h["step"]
    return None


def run_benchmark(benchmark, factorial, device="cpu"):
    print(f"\n{'=' * 72}\n{benchmark.upper()} — extended budget for full CG-PED\n{'=' * 72}")
    dataset = build_dataset(benchmark)

    naive_cell = factorial[benchmark]["cells"]["---"]
    full_cell = factorial[benchmark]["cells"]["CLD"]
    target = naive_cell["val_loss"]["mean"]
    naive_wall = naive_cell["wall_clock_sec"]["mean"]
    print(f"  target = naive mean val loss at {N_STEPS} steps: {target:.4f} "
          f"({naive_wall:.1f}s per run)")

    runs, samples = [], {}
    for seed in SEEDS:
        cfg = make_config("--D", seed, benchmark, dataset.cfg.max_len)
        teacher, _ = train_teacher(dataset, cfg, device=device)

        cfg = make_config("CLD", seed, benchmark, dataset.cfg.max_len)
        cfg.n_steps = EXTENDED_STEPS
        cfg.name = f"{benchmark}:CLD-extended:seed{seed}"
        res, model = run_training(dataset, cfg, teacher=teacher, device=device,
                                  return_model=True)

        hist = res["history"]
        catch_up = first_step_at_or_below(hist, target)
        best = min(h["val_loss"] for h in hist)
        # Wall clock is measured over the whole extended run; per-step cost is
        # near-constant, so this scales linearly and lets us ask what CG-PED
        # reaches inside the naive baseline's wall-clock budget.
        sec_per_step = res["wall_clock_sec"] / EXTENDED_STEPS
        steps_in_naive_budget = int(naive_wall / sec_per_step)
        at_equal_wall = min(
            (h for h in hist if h["step"] <= steps_in_naive_budget),
            key=lambda h: h["val_loss"], default=None)

        runs.append({
            "seed": seed,
            "final_val_loss": res["final_val_loss"],
            "best_val_loss": best,
            "catch_up_step": catch_up,
            "wall_clock_sec": res["wall_clock_sec"],
            "sec_per_step": round(sec_per_step, 4),
            "steps_within_naive_wall_clock": steps_in_naive_budget,
            "val_loss_at_equal_wall_clock": at_equal_wall["val_loss"] if at_equal_wall else None,
            "history": hist,
        })
        print(f"  seed={seed}: best {best:.4f} over {EXTENDED_STEPS} steps; "
              f"catch-up step {catch_up}; "
              f"at equal wall clock ({steps_in_naive_budget} steps) "
              f"{at_equal_wall['val_loss']:.4f}")

        if benchmark == "real" and seed == SEEDS[0]:
            samples["full_cgped"] = generate_sample(model, dataset, PROMPT, device=device)

    if benchmark == "real":
        cfg = make_config("---", SEEDS[0], benchmark, dataset.cfg.max_len)
        _, naive_model = run_training(dataset, cfg, device=device, return_model=True)
        samples["naive"] = generate_sample(naive_model, dataset, PROMPT, device=device)

    caught = [r["catch_up_step"] for r in runs if r["catch_up_step"] is not None]
    out = {
        "benchmark": benchmark,
        "target_naive_mean_val_loss": round(target, 4),
        "naive_steps": N_STEPS,
        "naive_wall_clock_sec": round(naive_wall, 2),
        "extended_steps": EXTENDED_STEPS,
        "seeds_that_caught_up": len(caught),
        "catch_up_steps": caught,
        "best_val_loss": summarise([r["best_val_loss"] for r in runs]),
        "equal_step_val_loss": full_cell["val_loss"],
        "equal_wall_clock_val_loss": summarise(
            [r["val_loss_at_equal_wall_clock"] for r in runs]),
        "cost_ratio_per_step": round(
            statistics.mean([r["sec_per_step"] for r in runs]) / (naive_wall / N_STEPS), 2),
        "runs": runs,
    }
    if samples:
        out["qualitative_samples"] = {"prompt": PROMPT, "temperature": 0.8, **samples}
    return out


def main():
    torch.set_num_threads(1)
    fpath = os.path.join(RESULTS_DIR, "factorial.json")
    if not os.path.exists(fpath):
        raise SystemExit("Run experiments/run_factorial.py first.")
    with open(fpath) as f:
        factorial = json.load(f)

    out = {b: run_benchmark(b, factorial) for b in ("synthetic", "real")}
    path = os.path.join(RESULTS_DIR, "budget.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'=' * 72}\nHEADLINE\n{'=' * 72}")
    for b in ("synthetic", "real"):
        o = out[b]
        print(f"\n{b}:")
        print(f"  naive @{N_STEPS} steps                {o['target_naive_mean_val_loss']:.4f}")
        print(f"  CG-PED @{N_STEPS} steps               {o['equal_step_val_loss']['mean']:.4f}")
        print(f"  CG-PED best over {EXTENDED_STEPS} steps       {o['best_val_loss']['mean']:.4f}")
        print(f"  CG-PED at equal wall clock       {o['equal_wall_clock_val_loss']['mean']:.4f}")
        print(f"  seeds reaching the naive target  {o['seeds_that_caught_up']}/{len(SEEDS)}"
              f"  {o['catch_up_steps']}")
        print(f"  cost per step vs naive           {o['cost_ratio_per_step']}x")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
