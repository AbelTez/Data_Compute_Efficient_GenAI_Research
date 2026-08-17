"""
plotting.py
-----------
Shared matplotlib helpers. `matplotlib.use("Agg")` MUST run before pyplot
is imported anywhere else in the process (headless sandbox, no display).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "a": "#2563eb",  # blue   - baseline / random / full-finetune / from-scratch
    "b": "#dc2626",  # red    - curriculum / lora / distilled / cg-ped
}


def plot_two_curves(hist_a, hist_b, label_a, label_b, title, ylabel, out_path,
                     metric="val_loss"):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xa = [h["step"] for h in hist_a]
    ya = [h[metric] for h in hist_a]
    xb = [h["step"] for h in hist_b]
    yb = [h[metric] for h in hist_b]
    ax.plot(xa, ya, color=COLORS["a"], marker="o", markersize=3, label=label_a, linewidth=2)
    ax.plot(xb, yb, color=COLORS["b"], marker="s", markersize=3, label=label_b, linewidth=2)
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def bar_compare(labels, values, title, ylabel, out_path, fmt="{:.0f}"):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    colors = [COLORS["a"], COLORS["b"]]
    bars = ax.bar(labels, values, color=colors[: len(labels)])
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                 ha="center", va="bottom", fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
