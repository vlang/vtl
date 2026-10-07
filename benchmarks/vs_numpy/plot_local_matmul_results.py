"""Render the committed Ryzen 9 5900X VTL vs NumPy matmul sample."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "assets"
OUTPUT.mkdir(parents=True, exist_ok=True)

sizes = [128, 256, 512]
results = {
    "f64": {
        "VTL pure V": [12.811, 23.364, 25.924],
        "NumPy": [59.799, 74.822, 96.765],
    },
    "f32": {
        "VTL pure V": [35.885, 53.213, 65.613],
        "NumPy": [189.172, 243.789, 276.393],
    },
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "svg.fonttype": "none",
    }
)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=False)
colors = {"VTL pure V": "#2563eb", "NumPy": "#475569"}
x = np.arange(len(sizes))
width = 0.34

for axis, (dtype, measurements) in zip(axes, results.items()):
    for offset, (label, values) in enumerate(measurements.items()):
        positions = x + (offset - 0.5) * width
        bars = axis.bar(positions, values, width, label=label, color=colors[label])
        axis.bar_label(bars, fmt="%.0f", padding=3, fontsize=8)
    axis.set_title(f"{dtype} matrix multiplication")
    axis.set_xticks(x, [f"{size}×{size}" for size in sizes])
    axis.set_xlabel("Matrix dimensions")
    axis.set_ylabel("GFLOPS (higher is better)")
    axis.grid(axis="y", color="#cbd5e1", alpha=0.55, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="upper left")

fig.suptitle("VTL vs NumPy · Ryzen 9 5900X · 2 threads", fontsize=15, weight="bold")
fig.text(
    0.5,
    0.015,
    "VTL -prod pure-V backend · NumPy 2.5.3 / scipy-openblas 0.3.34 · identical inputs",
    ha="center",
    color="#475569",
    fontsize=8,
)
fig.tight_layout(rect=(0, 0.055, 1, 0.92))
svg_path = OUTPUT / "matmul-ryzen-5900x.svg"
fig.savefig(svg_path, bbox_inches="tight")
svg_path.write_text(svg_path.read_text().replace(" \n", "\n"))
fig.savefig(OUTPUT / "matmul-ryzen-5900x.png", dpi=180, bbox_inches="tight")
