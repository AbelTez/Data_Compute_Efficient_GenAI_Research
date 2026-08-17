"""
E2 — Full Fine-Tuning vs. LoRA
================================
Isolates the effect of parameter-efficient adaptation. Both runs use the
SAME curriculum ordering (curriculum is held fixed "on" here so E2 is
measured on top of E1's better ordering, matching how CG-PED composes the
techniques) so the only variable is whether all weights are updated or
only low-rank adapters (r=4) inserted into attention/FFN projections.

Metrics of interest: trainable parameter count (the headline efficiency
number), wall-clock time per run, and final validation quality -- LoRA
should use far fewer trainable parameters for a small, acceptable quality
gap (or none, at this model scale).
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
                  use_curriculum=True, thresholds=(3.65, 3.6, 3.55, 3.5))

    rc_full = RunConfig(name="E2_full_finetune", use_lora=False, **common)
    rc_lora = RunConfig(name="E2_lora", use_lora=True, lora_r=4, lora_alpha=8, **common)

    res_full = run_training(ds, rc_full)
    res_lora = run_training(ds, rc_lora)

    out = {"full_finetune": res_full, "lora": res_lora}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "e2_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    os.makedirs(FIG_DIR, exist_ok=True)
    plot_two_curves(
        res_full["history"], res_lora["history"],
        "Full fine-tuning", "LoRA (r=4)",
        "E2: Validation Loss vs. Training Step",
        "Validation loss (next-token CE)",
        os.path.join(FIG_DIR, "e2_loss_curves.png"),
    )
    bar_compare(
        ["Full fine-tune", "LoRA (r=4)"],
        [res_full["trainable_params"], res_lora["trainable_params"]],
        "E2: Trainable Parameters", "# trainable params",
        os.path.join(FIG_DIR, "e2_trainable_params.png"),
    )

    reduction = 1 - res_lora["trainable_params"] / res_full["trainable_params"]
    print("=== E2: Full Fine-Tuning vs. LoRA ===")
    print(f"Full fine-tune -> trainable={res_full['trainable_params']}  "
          f"final_val_loss={res_full['final_val_loss']:.4f}  "
          f"wall_clock_s={res_full['wall_clock_sec']}")
    print(f"LoRA (r=4)     -> trainable={res_lora['trainable_params']}  "
          f"final_val_loss={res_lora['final_val_loss']:.4f}  "
          f"wall_clock_s={res_lora['wall_clock_sec']}")
    print(f"Trainable-parameter reduction: {reduction*100:.1f}%")
    return out


if __name__ == "__main__":
    main()
