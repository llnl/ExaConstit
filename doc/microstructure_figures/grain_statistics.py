"""Grain-size and neighbor statistics of several runs, as one figure and a printed table.

  python3 grain_statistics.py OUT.png LABEL=RUN_grains.csv [LABEL=RUN_grains.csv ...]

Reads the grain tables the generator writes (<basename>_grains.csv) and plots, for every run,
the distribution of the grain volume over the mean grain volume and of the number of neighbors
per grain. All grains are counted, including those cut by the outer surface. Needs numpy and
matplotlib.
"""
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Categorical colors in fixed order, with line styles as a second cue for color-blind readers
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
STYLES = ["-", "--", ":", "-."]
INK, MUTED = "#2b2b2b", "#8a8a85"


def read(path):
    """Volume and neighbor count of every grain in a <basename>_grains.csv."""
    with open(path) as table:
        rows = list(csv.DictReader(line for line in table if not line.startswith("#")))
    volume = np.array([float(r["volume"]) for r in rows])
    neighbors = np.array([int(r["num_neighbors"]) for r in rows])
    return volume, neighbors


out = sys.argv[1]
runs = [arg.split("=", 1) for arg in sys.argv[2:]]
plt.rcParams.update({"font.size": 11})
fig, (ax_v, ax_n) = plt.subplots(1, 2, figsize=(7.0, 2.8))
print("%-24s %6s %8s %8s %8s %8s" % ("run", "grains", "CV(V)", "V5/<V>", "V95/<V>", "<nbrs>"))
for k, (label, path) in enumerate(runs):
    volume, neighbors = read(path)
    v = volume / volume.mean()
    style = dict(color=COLORS[k], linestyle=STYLES[k], linewidth=2.0, label=label)
    ax_v.hist(v, bins=np.linspace(0.0, 4.0, 41), density=True, histtype="step", **style)
    ax_n.hist(neighbors, bins=np.arange(0.5, 40.5, 1.0), density=True, histtype="step", **style)
    print("%-24s %6d %8.2f %8.2f %8.2f %8.1f" % (label, v.size, v.std() / v.mean(),
                                                np.percentile(v, 5), np.percentile(v, 95),
                                                neighbors.mean()))
for ax, xlabel in ((ax_v, "grain volume / mean grain volume"),
                   (ax_n, "neighbors per grain")):
    ax.set_xlabel(xlabel, color=INK)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
ax_v.set_ylabel("fraction of grains", color=INK)
ax_v.legend(frameon=False, fontsize=9, labelcolor=INK)
fig.tight_layout()
fig.savefig(out, dpi=200, facecolor="white")
print("wrote", out)
