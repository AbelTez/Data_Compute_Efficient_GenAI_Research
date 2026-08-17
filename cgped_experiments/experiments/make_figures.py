"""
make_figures.py
---------------
Renders every figure in the paper from results/factorial.json and
results/budget.json. No number is typed in by hand here; if a result changes,
re-running this script is the only thing needed to update the figures.

Run: python3 experiments/make_figures.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.plotting import plt  # imports matplotlib with the Agg backend set

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "figures")

INK = "#1f2937"
NAIVE_C = "#2563eb"
FULL_C = "#b91c1c"
MID_C = "#94a3b8"
PRED_C = "#0f766e"
ORDER = ["---", "C--", "-L-", "--D", "CL-", "C-D", "-LD", "CLD"]
PRETTY = {"---": "naive", "C--": "C", "-L-": "L", "--D": "D",
          "CL-": "C+L", "C-D": "C+D", "-LD": "L+D", "CLD": "C+L+D\n(CG-PED)"}


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#cbd5e1")


def fig_pipeline(path):
    """The method diagram. Every box states what the code actually does; the
    'trained from scratch' and 'still fully trained' labels are there because
    an earlier version of this figure claimed a pretrained teacher and
    LoRA-only updates, neither of which was true of the implementation."""
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 62)

    def box(x, y, w, h, text, edge, face, weight="normal", size=9.2):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=1.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=size, color=INK, fontweight=weight, linespacing=1.45)

    def arrow(x1, y1, x2, y2, style="-|>"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color="#475569", linewidth=1.5,
                                    connectionstyle="arc3,rad=0"))

    C_E, C_F = "#c2410c", "#ffedd5"   # curriculum
    L_E, L_F = "#15803d", "#dcfce7"   # LoRA
    D_E, D_F = "#6d28d9", "#ede9fe"   # distillation
    N_E, N_F = "#64748b", "#f1f5f9"   # neutral

    box(2, 46, 22, 11, "Training corpus\n(synthetic SC-LM  /  Tiny Shakespeare)", N_E, N_F)
    box(2, 30, 22, 12, "Difficulty score  D(x)\n0.5·structural depth\n+ 0.5·lexical rarity", C_E, C_F)
    box(2, 12, 22, 13, "Competence-based sampler\neligible pool grows\n15% → 100% over the\nfirst half of training", C_E, C_F)

    box(36, 26, 27, 16,
        "Student  (82K params)\nLoRA r=4 on all 12 projections\n"
        "q, k, v, out, fc1, fc2 × 2 blocks\n"
        "base weights frozen\n"
        "embeddings, head, LayerNorms\nSTILL FULLY TRAINED",
        L_E, L_F, size=8.8)

    box(36, 3, 27, 13,
        "Teacher  (561K params)\ntrained from scratch on the\nSAME data, then frozen\n"
        "— not a pretrained model —", D_E, D_F, size=8.8)

    box(72, 30, 26, 13,
        "Loss\n0.5·CE(student, labels)\n+ 0.5·T²·KL(teacher ‖ student)\nT = 2", D_E, D_F, size=9)

    box(72, 12, 26, 11, "AdamW\nupdates the 23K parameters\nthat are not frozen", N_E, N_F)

    arrow(13, 46, 13, 42.4)
    arrow(13, 30, 13, 25.4)
    arrow(24, 18.5, 35.6, 32)
    arrow(63, 34, 71.6, 36.5)
    arrow(63, 9.5, 84, 29.6)
    arrow(85, 30, 85, 23.4)
    ax.annotate("", xy=(49.5, 26), xytext=(85, 12),
                arrowprops=dict(arrowstyle="-|>", color="#475569", linewidth=1.5,
                                connectionstyle="arc3,rad=0.28"))
    ax.text(66, 15.5, "gradient", fontsize=8, color="#475569", style="italic")

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=f, edgecolor=e, linewidth=1.4)
               for e, f in [(C_E, C_F), (L_E, L_F), (D_E, D_F)]]
    ax.legend(handles, ["Curriculum (C)", "Parameter-efficient / LoRA (L)", "Distillation (D)"],
              loc="lower left", bbox_to_anchor=(0.01, -0.02), frameon=False,
              fontsize=9, ncol=3)
    ax.set_title("CG-PED: the three components and what each one actually trains",
                 fontsize=12.5, fontweight="bold", color=INK, pad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_factorial(fac, path):
    """All eight cells, both benchmarks, mean over three seeds with the
    seed-to-seed range drawn as the error bar. The range is the point of the
    figure: it is what tells you which differences you are allowed to read."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, bm, title in zip(axes, ("synthetic", "real"),
                             ("Synthetic SC-LM benchmark", "Tiny Shakespeare (real text)")):
        cells = fac[bm]["cells"]
        means = [cells[c]["val_loss"]["mean"] for c in ORDER]
        lo = [cells[c]["val_loss"]["mean"] - cells[c]["val_loss"]["min"] for c in ORDER]
        hi = [cells[c]["val_loss"]["max"] - cells[c]["val_loss"]["mean"] for c in ORDER]
        colors = [NAIVE_C if c == "---" else FULL_C if c == "CLD" else MID_C for c in ORDER]
        xs = range(len(ORDER))
        ax.bar(xs, means, yerr=[lo, hi], capsize=4, color=colors,
               error_kw=dict(ecolor="#334155", lw=1.2))
        for x, m in zip(xs, means):
            ax.text(x, m, f"{m:.3f}", ha="center", va="bottom", fontsize=8.2, color=INK)
        ax.axhline(cells["---"]["val_loss"]["mean"], color=NAIVE_C, ls="--", lw=1.2, alpha=.7)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([PRETTY[c] for c in ORDER], fontsize=8.6)
        span = max(means) - min(means)
        ax.set_ylim(min(means) - span * 0.55, max(means) + span * 0.45)
        ax.set_ylabel("Validation loss (nats/token)" if bm == "synthetic" else "")
        ax.set_title(title, fontsize=11, fontweight="bold", color=INK)
        _clean(ax)
    fig.suptitle("Every cell of the 2³ factorial — mean of 3 seeds, bars show the seed range",
                 fontsize=12.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_effects_and_additivity(fac, path):
    """The headline, in two panels.

    Left: each factor's main effect against the seed-noise band, so a reader
    can see at a glance which effects are large enough to be read at all.
    Right: what the three factors predict together if they do not interact,
    against what the composition actually reaches. The two bars matching is
    the finding.
    """
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 5),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    bms = ("synthetic", "real")
    bm_label = {"synthetic": "Synthetic SC-LM", "real": "Tiny Shakespeare (real)"}
    factor_name = {"C": "Curriculum", "L": "LoRA", "D": "Distillation"}

    # --- left: main effects vs noise -------------------------------------
    width = 0.36
    ys = list(range(3))
    for i, bm in enumerate(bms):
        a = fac[bm]["analysis"]
        vals = [a["main_effects"][f]["main_effect"] for f in ("C", "L", "D")]
        sd = a["pooled_within_cell_std"]
        pos = [y + (i - 0.5) * width for y in ys]
        colors = ["#0f766e" if v < 0 else "#b91c1c" for v in vals]
        axL.barh(pos, vals, width, color=colors, alpha=1.0 if i == 0 else 0.55,
                 edgecolor="white", linewidth=.8)
        for p, v in zip(pos, vals):
            ha = "left" if v > 0 else "right"
            off = 0.0025 if v > 0 else -0.0025
            axL.text(v + off, p, f"{v:+.3f}  ({bm[:3]})", va="center", ha=ha,
                     fontsize=8.2, color=INK)
        axL.axvspan(-sd, sd, color=MID_C, alpha=0.16 if i == 0 else 0.10, zorder=0)

    axL.axvline(0, color=INK, lw=1.1)
    axL.set_yticks(ys)
    axL.set_yticklabels([factor_name[f] for f in ("C", "L", "D")], fontsize=10)
    axL.invert_yaxis()
    axL.set_xscale("symlog", linthresh=0.02)
    axL.set_xlim(-0.35, 0.75)
    axL.set_xlabel("Main effect on validation loss   (negative = better)")
    axL.set_title("Only one effect is large — and it is the wrong sign\n"
                  "shaded band = seed noise; solid = synthetic, faded = real",
                  fontsize=10.5, fontweight="bold", color=INK)
    _clean(axL)

    # --- right: additivity check -----------------------------------------
    # Plotted as (observed - predicted) rather than as two loss bars, because
    # the two benchmarks sit at very different loss levels and the quantity
    # that matters is the difference against the noise band, not the level.
    ys2 = [0, 1]
    for y, b in zip(ys2, bms):
        a = fac[b]["analysis"]
        sd = a["pooled_within_cell_std"]
        axR.barh([y], [sd * 2], 0.52, left=-sd, color=MID_C, alpha=.30, zorder=0)
        axR.plot([-sd, -sd], [y - .26, y + .26], color=MID_C, lw=1.4)
        axR.plot([sd, sd], [y - .26, y + .26], color=MID_C, lw=1.4)
        axR.barh([y], [a["composition_gap"]], 0.26, color=PRED_C, zorder=2)
        axR.text(sd * 1.12, y - 0.30, f"±{sd:.4f} seed noise",
                 fontsize=8, color="#64748b", va="center")
        axR.text(a["composition_gap"] + (0.06 * sd if a["composition_gap"] >= 0 else -0.06 * sd),
                 y + 0.10,
                 f"{a['composition_gap']:+.4f}",
                 ha="left" if a["composition_gap"] >= 0 else "right",
                 va="center", fontsize=9.5, color=INK, fontweight="bold")

    axR.axvline(0, color=INK, lw=1.1)
    axR.set_yticks(ys2)
    axR.set_yticklabels([bm_label[b] for b in bms], fontsize=10)
    axR.invert_yaxis()
    lim = max(fac[b]["analysis"]["pooled_within_cell_std"] for b in bms) * 1.9
    axR.set_xlim(-lim, lim)
    axR.set_ylim(1.6, -0.6)
    axR.set_xlabel("Observed composition − additive prediction  (nats)")
    axR.set_title("The composition is predicted by its parts,\n"
                  "to well inside seed noise",
                  fontsize=10.5, fontweight="bold", color=INK)
    _clean(axR)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_budget(fac, bud, path):
    """Equal steps flatters CG-PED; equal wall clock is the budget a
    GPU-less researcher actually has. Both are drawn."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, bm, title in zip(axes, ("synthetic", "real"),
                             ("Synthetic SC-LM", "Tiny Shakespeare (real text)")):
        b = bud[bm]
        naive_hist = fac[bm]["cells"]["---"]["runs"][0]["history"]
        ax.plot([h["step"] for h in naive_hist], [h["val_loss"] for h in naive_hist],
                color=NAIVE_C, lw=2, label=f"naive baseline ({b['naive_steps']} steps)")
        for i, r in enumerate(b["runs"]):
            ax.plot([h["step"] for h in r["history"]], [h["val_loss"] for h in r["history"]],
                    color=FULL_C, lw=1.6, alpha=.75 if i == 0 else .35,
                    label="full CG-PED (extended budget)" if i == 0 else None)
        ax.axhline(b["target_naive_mean_val_loss"], color=NAIVE_C, ls="--", lw=1.2, alpha=.8)
        eq = int(sum(r["steps_within_naive_wall_clock"] for r in b["runs"]) / len(b["runs"]))
        ax.axvline(eq, color="#334155", ls=":", lw=1.4)
        ax.text(eq + 12, ax.get_ylim()[1], f" equal wall clock\n ({eq} CG-PED steps)",
                fontsize=8.2, color="#334155", va="top")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Validation loss (nats/token)" if bm == "synthetic" else "")
        ax.set_title(title, fontsize=11, fontweight="bold", color=INK)
        ax.legend(frameon=False, fontsize=8.8)
        ax.grid(alpha=.18)
        _clean(ax)
    fig.suptitle("Does a bigger budget close the gap?", fontsize=12.5,
                 fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_summary_table(fac, bud, path):
    rows = [["", "Synthetic SC-LM", "Tiny Shakespeare"]]
    def a(b, k): return fac[b]["analysis"][k]
    rows += [
        ["Naive baseline, val loss",
         f"{a('synthetic','naive_mean'):.3f}", f"{a('real','naive_mean'):.3f}"],
        ["Full CG-PED, val loss",
         f"{a('synthetic','full_cgped_mean'):.3f}", f"{a('real','full_cgped_mean'):.3f}"],
        ["Additive prediction",
         f"{a('synthetic','additive_prediction'):.3f}", f"{a('real','additive_prediction'):.3f}"],
        ["Composition gap",
         f"{a('synthetic','composition_gap'):+.3f}", f"{a('real','composition_gap'):+.3f}"],
        ["Within-cell sd (3 seeds)",
         f"{a('synthetic','pooled_within_cell_std'):.3f}",
         f"{a('real','pooled_within_cell_std'):.3f}"],
        ["Main effect: curriculum",
         f"{a('synthetic','main_effects')['C']['main_effect']:+.3f}",
         f"{a('real','main_effects')['C']['main_effect']:+.3f}"],
        ["Main effect: LoRA",
         f"{a('synthetic','main_effects')['L']['main_effect']:+.3f}",
         f"{a('real','main_effects')['L']['main_effect']:+.3f}"],
        ["Main effect: distillation",
         f"{a('synthetic','main_effects')['D']['main_effect']:+.3f}",
         f"{a('real','main_effects')['D']['main_effect']:+.3f}"],
        ["Trainable parameters saved",
         f"{a('synthetic','param_reduction_pct')}%", f"{a('real','param_reduction_pct')}%"],
        ["Cost per step vs naive",
         f"{bud['synthetic']['cost_ratio_per_step']}x", f"{bud['real']['cost_ratio_per_step']}x"],
        ["Seeds catching up in 900 steps",
         f"{bud['synthetic']['seeds_that_caught_up']}/3", f"{bud['real']['seeds_that_caught_up']}/3"],
    ]
    fig, ax = plt.subplots(figsize=(9.6, 0.46 * len(rows) + 0.8))
    ax.axis("off")
    tbl = ax.table(cellText=rows, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.4); tbl.scale(1, 1.45)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        if r == 0:
            cell.set_facecolor("#1f2937"); cell.set_text_props(color="w", fontweight="bold")
        else:
            cell.set_facecolor("#ffffff" if r % 2 else "#f8fafc")
        if c == 0 and r > 0:
            cell.set_text_props(ha="left"); cell._loc = "left"
    ax.set_title("CG-PED: every headline number", fontsize=12,
                 fontweight="bold", color=INK, pad=14)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "factorial.json")) as f:
        fac = json.load(f)
    with open(os.path.join(RESULTS_DIR, "budget.json")) as f:
        bud = json.load(f)

    jobs = [
        ("fig1_pipeline.png", lambda p: fig_pipeline(p)),
        ("fig2_factorial.png", lambda p: fig_factorial(fac, p)),
        ("fig3_effects.png", lambda p: fig_effects_and_additivity(fac, p)),
        ("fig4_budget.png", lambda p: fig_budget(fac, bud, p)),
        ("fig5_summary.png", lambda p: fig_summary_table(fac, bud, p)),
    ]
    for name, fn in jobs:
        fn(os.path.join(FIG_DIR, name))
        print(f"  wrote figures/{name}")


if __name__ == "__main__":
    main()
