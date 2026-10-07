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
        "VTL pure V": [15.437, 21.056, 26.238],
        "VTL CBLAS": [43.101, 86.799, 111.410],
        "NumPy": [56.544, 73.933, 100.248],
    },
    "f32": {
        "VTL pure V": [29.668, 46.374, 59.277],
        "VTL CBLAS": [127.884, 175.969, 239.913],
        "NumPy": [84.029, 196.316, 228.397],
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
colors = {"VTL pure V": "#94a3b8", "VTL CBLAS": "#2563eb", "NumPy": "#475569"}
x = np.arange(len(sizes))
width = 0.25

for axis, (dtype, measurements) in zip(axes, results.items()):
    for offset, (label, values) in enumerate(measurements.items()):
        positions = x + (offset - 1) * width
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
    "VTL -prod pure V / CBLAS · NumPy 2.5.3 · identical inputs · 2 threads",
    ha="center",
    color="#475569",
    fontsize=8,
)
fig.tight_layout(rect=(0, 0.055, 1, 0.92))
svg_path = OUTPUT / "matmul-ryzen-5900x.svg"
fig.savefig(svg_path, bbox_inches="tight")
svg_path.write_text(svg_path.read_text().replace(" \n", "\n"))
fig.savefig(OUTPUT / "matmul-ryzen-5900x.png", dpi=180, bbox_inches="tight")
