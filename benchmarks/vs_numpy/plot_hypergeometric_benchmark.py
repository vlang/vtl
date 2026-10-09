"""Generate the checked-in VTL/NumPy hypergeometric benchmark figure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


OUTPUT_DIR = Path(__file__).resolve().parents[2] / "docs" / "assets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

labels = ["VTL · -prod", "NumPy · default_rng"]
times_ms = [3.686, 8.545]
colors = ["#147d64", "#52718d"]

fig, ax = plt.subplots(figsize=(9, 4.8), dpi=180)
fig.patch.set_facecolor("#f5f7fa")
ax.set_facecolor("#f5f7fa")
bars = ax.barh(labels, times_ms, color=colors, height=0.48, zorder=3)
ax.invert_yaxis()
ax.set_xlim(0, 10.4)
ax.set_xlabel("Mean time for 100,000 variates (ms)", labelpad=12, color="#344054")
fig.suptitle(
    "Hypergeometric sampling",
    x=0.24,
    y=0.97,
    ha="left",
    fontsize=18,
    weight="bold",
    color="#172b4d",
)
fig.text(
    0.24,
    0.88,
    "ngood=50,000 · nbad=50,000 · nsample=50,000 · 2 warmups · 7 timed calls",
    fontsize=9,
    color="#52606d",
)
ax.xaxis.grid(True, color="#d9e0e7", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.spines["bottom"].set_color("#aab7c4")
ax.tick_params(axis="y", length=0, labelsize=11, colors="#172b4d", pad=10)
ax.tick_params(axis="x", colors="#52606d")
for bar, value in zip(bars, times_ms):
    ax.text(value + 0.18, bar.get_y() + bar.get_height() / 2, f"{value:.3f} ms", va="center", fontsize=11, weight="bold", color="#172b4d")
fig.text(
    0.24,
    0.035,
    "2.32× faster in this local sample · Ryzen 9 5900X · V 0.5.2 · NumPy 2.5.3",
    ha="left",
    fontsize=8,
    color="#52606d",
)
fig.subplots_adjust(left=0.24, right=0.97, top=0.76, bottom=0.23)
fig.savefig(OUTPUT_DIR / "hypergeometric-performance.svg", bbox_inches="tight", facecolor=fig.get_facecolor())
fig.savefig(OUTPUT_DIR / "hypergeometric-performance.png", bbox_inches="tight", facecolor=fig.get_facecolor())
