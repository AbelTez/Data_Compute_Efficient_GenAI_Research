"""
E1 — Random vs. Curriculum (Structure- and Rarity-Informed) Data Ordering
===========================================================================
Isolates the effect of *ordering alone*: identical model, identical
optimizer, identical number of gradient steps and batch size. The only
difference is whether training examples are drawn uniformly at random
each step, or via a competence-based pacing schedule that exposes the
model to easy (low structural-depth, high-frequency-token) examples
first.

Hypothesis (Sec 3.2 of the paper): curriculum ordering should reach a
sufficiently tight validation-loss threshold in fewer steps than random
ordering, because late training is not "wasted" re-learning basic
structure that easy examples could have taught earlier. Note this
benefit is expected to show up at tighter (harder-to-reach) thresholds
specifically: competence-based pacing deliberately restricts early
exposure to the easiest examples, which can make it slightly slower to
reach very loose thresholds even while being clearly faster to reach
tight ones -- both effects are reported honestly below.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import SyntheticConfig, SyntheticCompositionalDataset
from src.trainer import RunConfig, run_training
from src.plotting import plot_two_curves, bar_compare

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def main():
    cfg_data = SyntheticConfig()
    ds = SyntheticCompositionalDataset(cfg_data)

    common = dict(n_steps=300, eval_every=10, seed=7, lr=4e-3,
                  thresholds=(3.65, 3.6, 3.55, 3.5))

    rc_random = RunConfig(name="E1_random_order", use_curriculum=False, **common)
    rc_curric = RunConfig(name="E1_curriculum_order", use_curriculum=True, **common)

    res_random = run_training(ds, rc_random)
    res_curric = run_training(ds, rc_curric)

    out = {"random": res_random, "curriculum": res_curric}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "e1_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    os.makedirs(FIG_DIR, exist_ok=True)
    plot_two_curves(
        res_random["history"], res_curric["history"],
        "Random order", "Curriculum order (structure + rarity)",
        "E1: Validation Loss vs. Training Step",
        "Validation loss (next-token CE)",
        os.path.join(FIG_DIR, "e1_loss_curves.png"),
    )

    steps_r = steps_c = None
    th_used = None
    for th in (3.5, 3.55, 3.6, 3.65):
        sr = res_random["steps_to_threshold"].get(str(th)) or res_random["steps_to_threshold"].get(th)
        sc = res_curric["steps_to_threshold"].get(str(th)) or res_curric["steps_to_threshold"].get(th)
        if sr is not None and sc is not None:
            steps_r, steps_c, th_used = sr, sc, th
            break
    if steps_r is not None and steps_c is not None:
        bar_compare(
            ["Random", "Curriculum"], [steps_r, steps_c],
            f"E1: Steps to reach val_loss ≤ {th_used}", "Training steps",
            os.path.join(FIG_DIR, "e1_steps_to_threshold.png"),
        )

    print("=== E1: Random vs. Curriculum Ordering ===")
    print(f"Random    -> final val_loss={res_random['final_val_loss']:.4f}  "
          f"val_ppl={res_random['final_val_ppl']:.2f}  "
          f"steps_to_{th_used}={steps_r}")
    print(f"Curriculum-> final val_loss={res_curric['final_val_loss']:.4f}  "
          f"val_ppl={res_curric['final_val_ppl']:.2f}  "
          f"steps_to_{th_used}={steps_c}")
    if steps_r and steps_c:
        direction = "faster" if steps_c < steps_r else "slower"
        print(f"Curriculum reaches threshold {th_used} "
              f"{abs(1 - steps_c/steps_r)*100:.1f}% {direction} "
              f"({steps_c} vs {steps_r} steps)")
    print(f"Full steps-to-threshold tables -- random: {res_random['steps_to_threshold']}, "
          f"curriculum: {res_curric['steps_to_threshold']}")
    return out


if __name__ == "__main__":
    main()
