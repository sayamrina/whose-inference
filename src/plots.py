"""Figures for Part 4 (results/mech.json -> results/*.png)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
R = json.loads((ROOT / "results" / "mech.json").read_text())
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


def patching():
    rec = R["patching"]["recovery"]
    layers = list(range(len(rec["final"])))
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=160)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.plot(layers, rec["kb_token"], color=BLUE, lw=2, marker="o", ms=4)
    ax.plot(layers, rec["final"], color=ORANGE, lw=2, marker="o", ms=4)
    ax.text(2, 1.06, "patched at the PKB entry token", color=INK, fontsize=9)
    ax.text(9.5, 0.12, "patched at the answer position", color=INK, fontsize=9)
    ax.set_xlabel("Layer (Qwen2.5-0.5B-Instruct, 24 layers)")
    ax.set_ylabel("Share of PKB effect restored")
    ax.set_ylim(-0.1, 1.2)
    ax.set_title(f"The user's belief moves from the PKB entry to the answer in layers 16-22 (n = {R['patching']['n']})",
                 loc="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "patching.png")


def steering():
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=160)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for (layer, d), color in zip(sorted(R["steering"].items(), key=lambda x: int(x[0])), [BLUE, ORANGE]):
        xs = [r["alpha"] for r in d["results"]]
        ys = [100 * r["answer_is_user_belief"] for r in d["results"]]
        ax.plot(xs, ys, color=color, lw=2, marker="o", ms=5)
        ax.text(xs[-1] + 0.4, ys[-1], f"layer {layer}", color=INK, fontsize=9, va="center")
    ax.set_xlabel("Steering strength α (added along the 'user perspective' direction)")
    ax.set_ylabel("World question answered\nwith the user's belief (%)")
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.5, 19)
    controls = [r["no_pkb_still_true_capital"] for d in R["steering"].values() for r in d["results"]]
    ax.set_title("One direction switches the answer perspective (test half, n = 24)", loc="left", fontsize=10.5, color=INK)
    ax.text(0, 4, f"Control without PKB: true capital kept in ≥ {100 * min(controls):.0f}% of prompts at every α",
            color=MUTED, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "steering.png")


if __name__ == "__main__":
    patching()
    steering()
