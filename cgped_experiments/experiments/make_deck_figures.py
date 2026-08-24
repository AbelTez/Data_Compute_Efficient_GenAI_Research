import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

INK   = "#16202E"
MUTED = "#64748B"
RED   = "#C0392B"
TEAL  = "#0E8074"
AMBER = "#E08A2E"
BLUE  = "#2563EB"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    os.pardir, "figures") + os.sep

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#CBD5E1",
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
})

OUT = "/tmp/claude-1000/-home-abel-Desktop-gheero-group-AI-Research-CG-PED-Data-Compute-Efficient-GenAI-Research/412238af-7030-4189-b30b-958cec67435b/scratchpad/deck/"

# ---------------------------------------------------------------- transfer fig
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.4), dpi=170,
                               gridspec_kw={"width_ratios": [1.0, 1.15]})

# left: LoRA's price tag, scratch vs pretrained
labels = ["Trained from\nscratch", "Pretrained\nbackbone"]
vals   = [0.286, 0.082]
bars = ax1.bar(labels, vals, width=0.52, color=[RED, "#E4897F"], zorder=3)
ax1.axhline(0.022, ls="--", lw=1.6, color=MUTED, zorder=2)
ax1.text(0.5, 0.029, "seed noise 0.022", ha="center", va="bottom",
         fontsize=9.5, color=MUTED)
for b, v in zip(bars, vals):
    ax1.text(b.get_x() + b.get_width()/2, v + 0.008, f"+{v:.3f}", ha="center",
             va="bottom", fontsize=15, fontweight="bold", color=INK)
ax1.annotate("", xy=(0.98, 0.105), xytext=(0.30, 0.318),
             arrowprops=dict(arrowstyle="-|>", lw=2.2, color=TEAL,
                             connectionstyle="arc3,rad=-0.28"))
ax1.text(0.62, 0.283, "71% smaller", fontsize=15, fontweight="bold",
         color=TEAL, ha="center")
ax1.set_ylim(0, 0.35)
ax1.set_ylabel("LoRA's cost vs full fine-tuning  (nats)", fontsize=11.5)
ax1.set_title("LoRA's price tag shrinks — but never reaches zero",
              fontsize=13.5, fontweight="bold", pad=14)
ax1.spines[["top", "right"]].set_visible(False)
ax1.grid(axis="y", color="#EEF2F6", zorder=0)
ax1.tick_params(labelsize=11.5)

# right: where the loss actually lands on Austen
names = ["Backbone alone\n(no adaptation)", "From scratch\nfull fine-tune",
         "Pretrained\nadapters only", "Pretrained\nLoRA", "Pretrained\nfull fine-tune"]
loss  = [2.111, 1.912, 1.723, 1.650, 1.568]
cols  = [MUTED, BLUE, "#E4897F", RED, TEAL]
y = np.arange(len(names))[::-1]
ax2.barh(y, loss, height=0.58, color=cols, zorder=3)
for yy, v in zip(y, loss):
    ax2.text(v + 0.018, yy, f"{v:.3f}", va="center", fontsize=12.5,
             fontweight="bold", color=INK)
ax2.set_yticks(y); ax2.set_yticklabels(names, fontsize=11)
ax2.set_xlim(1.4, 2.32)
ax2.set_xlabel("Validation loss on Jane Austen  (lower is better)", fontsize=11.5)
ax2.set_title("Shakespeare → Austen: a real distribution shift",
              fontsize=13.5, fontweight="bold", pad=14)
ax2.spines[["top", "right", "left"]].set_visible(False)
ax2.grid(axis="x", color="#EEF2F6", zorder=0)
ax2.tick_params(axis="x", labelsize=11)

fig.tight_layout(pad=1.4)
fig.savefig(OUT + "fig_transfer.png", facecolor="white")
print("wrote fig_transfer.png")

# ------------------------------------------------------- three worlds fig
fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.0), dpi=170)
worlds = [
    ("SYNERGY",      "the whole beats the sum", [1, 1, 1, 3.9], TEAL),
    ("ADDITIVITY",   "the whole equals the sum", [1, 1, 1, 3.0], BLUE),
    ("INTERFERENCE", "the whole loses to the sum", [1, 1, 1, 1.9], RED),
]
for ax, (title, sub, v, col) in zip(axes, worlds):
    xs = ["C", "L", "D", "C+L+D"]
    cols = [AMBER, RED, TEAL, col]
    ax.bar(xs, v, color=cols, width=0.6, zorder=3)
    ax.axhline(3.0, ls="--", lw=1.5, color=MUTED, zorder=2)
    ax.text(3.0, v[3] + 0.12, {"SYNERGY": "more", "ADDITIVITY": "exactly",
                               "INTERFERENCE": "less"}[title],
            ha="center", fontsize=11.5, fontweight="bold", color=col)
    ax.set_ylim(0, 5.0)
    ax.set_title(title, fontsize=13, fontweight="bold", color=col, pad=8)
    ax.text(0.5, -0.30, sub, transform=ax.transAxes, ha="center",
            fontsize=11, color=MUTED)
    ax.set_yticks([]); ax.tick_params(labelsize=11.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
axes[0].text(-0.05, 3.05, "sum of the parts", fontsize=9.5, color=MUTED)
fig.tight_layout(pad=1.2)
fig.subplots_adjust(bottom=0.22)
fig.savefig(OUT + "fig_worlds.png", facecolor="white")
print("wrote fig_worlds.png")
