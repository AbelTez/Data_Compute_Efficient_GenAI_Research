"""
make_architecture_diagram.py
-----------------------------
Renders Figure 1 of the paper: the CG-PED pipeline architecture, showing
how the three components (curriculum-guided data ordering, LoRA adaptation,
distillation from a frozen teacher) compose into a single training loop.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import os

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def box(ax, xy, w, h, text, fc="#eff6ff", ec="#2563eb", fontsize=9.5, fontweight="normal"):
    b = FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.6, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(b)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
             fontsize=fontsize, fontweight=fontweight, wrap=True)
    return b


def arrow(ax, p1, p2, color="#334155", style="-|>", lw=1.6, connectionstyle=None):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14,
                         color=color, linewidth=lw,
                         connectionstyle=connectionstyle)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(5.5, 7.7, "CG-PED: Curriculum-Guided Parameter-Efficient Distillation",
            ha="center", fontsize=14, fontweight="bold")

    # Raw corpus
    box(ax, (0.3, 6.4), 2.2, 0.9, "Raw Text\nCorpus", fc="#f8fafc", ec="#64748b")

    # Morphology-informed difficulty scorer
    box(ax, (0.3, 4.9), 2.6, 1.1,
        "Difficulty Scorer\n(structural depth + lexical rarity)",
        fc="#fef3c7", ec="#d97706")

    # Curriculum sampler
    box(ax, (0.3, 3.3), 2.6, 1.1,
        "Curriculum Sampler\n(competence-based pacing:\neasy \u2192 hard)",
        fc="#fef3c7", ec="#d97706")

    arrow(ax, (1.4, 6.4), (1.4, 6.0))
    arrow(ax, (1.4, 4.9), (1.4, 4.4))

    # Teacher model (frozen)
    box(ax, (0.3, 1.2), 2.6, 1.4,
        "Teacher LM\n(larger, pretrained /\nfully fine-tuned)\n[frozen]",
        fc="#ede9fe", ec="#7c3aed")

    # Student model with LoRA
    box(ax, (4.3, 3.0), 2.9, 1.6,
        "Student LM\n+ LoRA adapters\n(base weights frozen,\nonly A, B trained)",
        fc="#dcfce7", ec="#16a34a")

    arrow(ax, (2.9, 3.85), (4.3, 3.8))  # curriculum sampler -> student batches
    arrow(ax, (1.6, 1.9), (4.3, 3.4), connectionstyle="arc3,rad=0.25")  # teacher -> student (soft labels)

    # Loss
    box(ax, (7.9, 3.0), 2.7, 1.7,
        "Combined Loss\n\u03b1\u00b7CE(student, labels) +\n(1-\u03b1)\u00b7T\u00b2\u00b7KL(teacher \u2016 student)",
        fc="#fee2e2", ec="#dc2626")
    arrow(ax, (7.2, 3.8), (7.9, 3.8))
    arrow(ax, (1.6, 1.9), (7.9, 3.3), connectionstyle="arc3,rad=0.45")

    # Optimizer / backward
    box(ax, (7.9, 0.9), 2.7, 1.3,
        "AdamW\n(updates LoRA params\nonly)", fc="#f1f5f9", ec="#475569")
    arrow(ax, (9.2, 3.0), (9.2, 2.2))
    arrow(ax, (9.2, 0.9), (9.2, 0.5))
    arrow(ax, (9.2, 0.5), (5.75, 0.5), connectionstyle="arc3,rad=0")
    arrow(ax, (5.75, 0.5), (5.75, 3.0))

    # Output
    box(ax, (4.3, 5.4), 2.9, 1.1,
        "Efficient Small\nGenerative Language Model",
        fc="#e0f2fe", ec="#0284c7", fontweight="bold")
    arrow(ax, (5.75, 4.6), (5.75, 5.4))

    legend_elems = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#fef3c7",
               markeredgecolor="#d97706", markersize=14, label="Curriculum component"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#dcfce7",
               markeredgecolor="#16a34a", markersize=14, label="Parameter-efficient (LoRA) component"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#ede9fe",
               markeredgecolor="#7c3aed", markersize=14, label="Distillation component (frozen teacher)"),
    ]
    ax.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, -0.02),
              ncol=1, frameon=False, fontsize=9.5)

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "architecture_diagram.png"), dpi=170)
    plt.close(fig)
    print("Saved architecture_diagram.png")


if __name__ == "__main__":
    main()
