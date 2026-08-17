"""
E3 — From-Scratch Student vs. Knowledge-Distilled Student
============================================================
A larger "teacher" model is trained first with a plain recipe (random
order, full fine-tune, no distillation -- it has nothing to distill
from). Then two identically-sized "student" models are trained under
curriculum ordering: one purely on hard-label next-token CE, the other
with the standard KD loss (soft teacher targets + hard labels, Hinton et
al. 2015) using the same teacher's logits.

Hypothesis: distillation should transfer some of the teacher's better
calibrated distribution over plausible continuations, helping the small
student converge faster / reach lower validation loss than training the
same-sized student from scratch under an identical budget.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import SyntheticConfig, SyntheticCompositionalDataset
from src.trainer import RunConfig, run_training, train_teacher
from src.plotting import plot_two_curves, bar_compare

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def main():
    cfg_data = SyntheticConfig()
    ds = SyntheticCompositionalDataset(cfg_data)

    teacher_cfg = RunConfig(name="E3_teacher", seed=7, teacher_steps=250, lr=3e-3)
    teacher, teacher_info = train_teacher(ds, teacher_cfg)

    common = dict(n_steps=300, eval_every=10, seed=7, lr=4e-3,
                  use_curriculum=True, thresholds=(3.65, 3.6, 3.55, 3.5))

    rc_scratch = RunConfig(name="E3_from_scratch", use_distillation=False, **common)
    rc_distill = RunConfig(name="E3_distilled", use_distillation=True,
                            kd_temperature=2.0, kd_alpha=0.5, **common)

    res_scratch = run_training(ds, rc_scratch)
    res_distill = run_training(ds, rc_distill, teacher=teacher)

    out = {"teacher": teacher_info, "from_scratch": res_scratch, "distilled": res_distill}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "e3_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    os.makedirs(FIG_DIR, exist_ok=True)
    plot_two_curves(
        res_scratch["history"], res_distill["history"],
        "From scratch", "Distilled (KD from teacher)",
        "E3: Validation Loss vs. Training Step",
        "Validation loss (next-token CE)",
        os.path.join(FIG_DIR, "e3_loss_curves.png"),
    )
    bar_compare(
        ["From scratch", "Distilled"],
        [res_scratch["final_val_ppl"], res_distill["final_val_ppl"]],
        "E3: Final Validation Perplexity", "Perplexity",
        os.path.join(FIG_DIR, "e3_final_ppl.png"), fmt="{:.1f}",
    )

    print("=== E3: From-Scratch vs. Distilled Student ===")
    print(f"Teacher        -> val_loss={teacher_info['val_loss']:.4f}  "
          f"val_ppl={teacher_info['val_ppl']:.2f}  "
          f"params={sum(p.numel() for p in teacher.parameters())}")
    print(f"From scratch   -> final_val_loss={res_scratch['final_val_loss']:.4f}  "
          f"val_ppl={res_scratch['final_val_ppl']:.2f}")
    print(f"Distilled      -> final_val_loss={res_distill['final_val_loss']:.4f}  "
          f"val_ppl={res_distill['final_val_ppl']:.2f}")
    return out


if __name__ == "__main__":
    main()
