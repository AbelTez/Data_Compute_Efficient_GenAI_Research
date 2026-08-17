"""
make_timeline.py
------------------
Renders the 4-week research execution timeline referenced in the paper's
"Research Plan" section.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

TASKS = [
    ("Lit. review & gap analysis", 0, 4, "#94a3b8"),
    ("Baseline reproduction\n(curriculum, LoRA, KD individually)", 1, 4, "#94a3b8"),
    ("CG-PED method design", 3, 5, "#2563eb"),
    ("Synthetic compositional (SC-LM)\nbenchmark build", 4, 6, "#2563eb"),
    ("Core pipeline implementation\n(model, LoRA, KD, curriculum)", 5, 9, "#16a34a"),
    ("E1: curriculum vs. random order", 8, 10, "#d97706"),
    ("E2: LoRA vs. full fine-tune", 9, 11, "#d97706"),
    ("E3: distilled vs. from-scratch", 10, 12, "#d97706"),
    ("E4: naive vs. full CG-PED", 11, 14, "#d97706"),
    ("Real-corpus sourcing (public domain)", 13, 15, "#0284c7"),
    ("E5: real-corpus validation", 14, 17, "#0284c7"),
    ("Results analysis & figures", 16, 19, "#7c3aed"),
    ("Paper drafting", 17, 22, "#dc2626"),
    ("Internal review & revision", 21, 23, "#dc2626"),
    ("Final submission package", 23, 24, "#dc2626"),
]

WEEK_BOUNDARIES = [0, 6, 12, 18, 24]
WEEK_LABELS = ["Week 1", "Week 2", "Week 3", "Week 4"]


def main():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    for i, (name, start, end, color) in enumerate(TASKS):
        y = len(TASKS) - i
        ax.barh(y, end - start, left=start, height=0.55, color=color, alpha=0.9)
        ax.text(-0.3, y, name, ha="right", va="center", fontsize=8.5)

    for wb in WEEK_BOUNDARIES:
        ax.axvline(wb, color="#cbd5e1", linewidth=1, zorder=0)
    for i, wl in enumerate(WEEK_LABELS):
        mid = (WEEK_BOUNDARIES[i] + WEEK_BOUNDARIES[i + 1]) / 2
        ax.text(mid, len(TASKS) + 0.8, wl, ha="center", fontsize=10, fontweight="bold")

    ax.set_xlim(0, 24)
    ax.set_ylim(0, len(TASKS) + 1.6)
    ax.set_yticks([])
    ax.set_xticks(WEEK_BOUNDARIES)
    ax.set_xticklabels([f"Day {d}" for d in WEEK_BOUNDARIES])
    ax.set_xlabel("Days into the 4-week research assignment")
    ax.set_title("CG-PED Research Execution Timeline (4 Weeks)",
                  fontsize=13, fontweight="bold")
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "timeline_gantt.png"), dpi=170)
    plt.close(fig)
    print("Saved timeline_gantt.png")


if __name__ == "__main__":
    main()
