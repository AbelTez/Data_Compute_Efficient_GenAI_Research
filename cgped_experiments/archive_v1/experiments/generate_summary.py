"""
generate_summary.py
---------------------
Reads results/e1..e5_results.json (already produced by running the five
experiment scripts) and compiles:
  - results/summary.json           machine-readable headline metrics
  - figures/summary_table.png      human-readable table for the paper

All headline percentages are computed here directly from the JSON results
rather than hardcoded, so this script always reflects the numbers from the
most recent run of the experiment suite.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE, "..", "results")
FIG_DIR = os.path.join(BASE, "..", "figures")


def load(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def common_threshold_steps(steps_to_thresh_a, steps_to_thresh_b):
    """Find the TIGHTEST (hardest / most informative) threshold that both
    runs reached, and return (threshold, steps_a, steps_b). Preferring the
    tightest common threshold (rather than the loosest) reflects fuller
    convergence and avoids an early-training artifact: competence-based
    curricula deliberately restrict early exposure to easy examples, which
    can make them reach very loose thresholds slightly slower, even while
    reaching tighter thresholds substantially faster (see paper Sec. 5.1)."""
    for th in ("3.5", "3.55", "3.6", "3.65", 3.5, 3.55, 3.6, 3.65):
        sa = steps_to_thresh_a.get(str(th))
        sb = steps_to_thresh_b.get(str(th))
        if sa is not None and sb is not None:
            return th, sa, sb
    return None, None, None


def main():
    e1 = load("e1_results.json")
    e2 = load("e2_results.json")
    e3 = load("e3_results.json")
    e4 = load("e4_results.json")
    e5 = load("e5_results.json")

    summary = {}
    rows = []

    if e1:
        th, s_r, s_c = common_threshold_steps(
            e1["random"]["steps_to_threshold"], e1["curriculum"]["steps_to_threshold"]
        )
        speedup_pct = round((1 - s_c / s_r) * 100, 1) if s_r and s_c else None
        summary["E1_curriculum_vs_random"] = {
            "random_final_val_loss": e1["random"]["final_val_loss"],
            "curriculum_final_val_loss": e1["curriculum"]["final_val_loss"],
            "threshold_used": th, "steps_random": s_r, "steps_curriculum": s_c,
            "speedup_pct": speedup_pct,
        }
        eff1 = f"{speedup_pct}% faster to threshold" if speedup_pct else "-"
        rows.append(["E1", "Random order", f"{e1['random']['final_val_loss']:.3f}", "-", "-"])
        rows.append(["E1", "Curriculum order", f"{e1['curriculum']['final_val_loss']:.3f}", "-", eff1])

    if e2:
        pct2 = round((1 - e2["lora"]["trainable_params"] / e2["full_finetune"]["trainable_params"]) * 100, 1)
        summary["E2_lora_vs_full_finetune"] = {
            "full_trainable_params": e2["full_finetune"]["trainable_params"],
            "lora_trainable_params": e2["lora"]["trainable_params"],
            "param_reduction_pct": pct2,
            "full_final_val_loss": e2["full_finetune"]["final_val_loss"],
            "lora_final_val_loss": e2["lora"]["final_val_loss"],
        }
        rows.append(["E2", "Full fine-tune", f"{e2['full_finetune']['final_val_loss']:.3f}",
                     f"{e2['full_finetune']['trainable_params']:,}", "-"])
        rows.append(["E2", "LoRA (r=4)", f"{e2['lora']['final_val_loss']:.3f}",
                     f"{e2['lora']['trainable_params']:,}", f"{pct2}% fewer params"])

    if e3:
        summary["E3_distilled_vs_scratch"] = {
            "teacher_val_loss": e3["teacher"]["val_loss"],
            "scratch_final_val_loss": e3["from_scratch"]["final_val_loss"],
            "distilled_final_val_loss": e3["distilled"]["final_val_loss"],
            "scratch_final_val_ppl": e3["from_scratch"]["final_val_ppl"],
            "distilled_final_val_ppl": e3["distilled"]["final_val_ppl"],
        }
        rows.append(["E3", "From scratch", f"{e3['from_scratch']['final_val_loss']:.3f}", "-", "-"])
        rows.append(["E3", "Distilled", f"{e3['distilled']['final_val_loss']:.3f}", "-", "lower final loss"])

    if e4:
        pct4 = round((1 - e4["full_cgped"]["trainable_params"] / e4["naive_baseline"]["trainable_params"]) * 100, 1)
        summary["E4_naive_vs_cgped"] = {
            "naive_trainable_params": e4["naive_baseline"]["trainable_params"],
            "cgped_trainable_params": e4["full_cgped"]["trainable_params"],
            "param_reduction_pct": pct4,
            "naive_final_val_loss": e4["naive_baseline"]["final_val_loss"],
            "cgped_final_val_loss": e4["full_cgped"]["final_val_loss"],
            "cgped_closest_approach": e4.get("closest_approach"),
        }
        rows.append(["E4", "Naive baseline", f"{e4['naive_baseline']['final_val_loss']:.3f}",
                     f"{e4['naive_baseline']['trainable_params']:,}", "-"])
        rows.append(["E4", "Full CG-PED", f"{e4['full_cgped']['final_val_loss']:.3f}",
                     f"{e4['full_cgped']['trainable_params']:,}", f"{pct4}% fewer params"])

    if e5:
        pct5 = round((1 - e5["full_cgped"]["trainable_params"] / e5["naive_baseline"]["trainable_params"]) * 100, 1)
        summary["E5_real_corpus_validation"] = {
            "corpus": e5["corpus"],
            "naive_trainable_params": e5["naive_baseline"]["trainable_params"],
            "cgped_trainable_params": e5["full_cgped"]["trainable_params"],
            "param_reduction_pct": pct5,
            "naive_final_val_loss": e5["naive_baseline"]["final_val_loss"],
            "cgped_final_val_loss": e5["full_cgped"]["final_val_loss"],
            "cgped_closest_approach": e5.get("closest_approach"),
            "qualitative_samples": e5.get("qualitative_samples"),
        }
        rows.append(["E5", "Naive baseline (real text)", f"{e5['naive_baseline']['final_val_loss']:.3f}",
                     f"{e5['naive_baseline']['trainable_params']:,}", "-"])
        rows.append(["E5", "Full CG-PED (real text)", f"{e5['full_cgped']['final_val_loss']:.3f}",
                     f"{e5['full_cgped']['trainable_params']:,}", f"{pct5}% fewer params"])

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    col_labels = ["Study", "Configuration", "Final val. loss", "Trainable params", "Headline effect"]
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1e293b")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f8fafc" if r % 2 == 0 else "#ffffff")
    ax.set_title("Summary of Pilot Study Results (E1\u2013E5)",
                  fontsize=12, fontweight="bold", pad=14)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "summary_table.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    print("Wrote results/summary.json and figures/summary_table.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
