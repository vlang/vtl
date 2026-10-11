"""NumPy reference for the VTL N-dimensional histogram example."""

import numpy as np


sample = np.array(
    [[0.1, 0.2, 0.3], [0.8, 0.7, 0.9], [1.0, 1.0, 1.0]], dtype=np.float64
)
edges = ([0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
weights = np.array([1.0, 2.0, 3.0], dtype=np.float64)

counts, _ = np.histogramdd(sample, bins=edges)
density, _ = np.histogramdd(sample, bins=edges, density=True)
weighted_density, _ = np.histogramdd(sample, bins=edges, weights=weights, density=True)

print(f"NumPy {np.__version__} histogramdd count dtype={counts.dtype}")
print(f"Counts: {counts.ravel().tolist()}")
print(f"Density: {density.ravel().tolist()}")
print(f"Weighted density: {weighted_density.ravel().tolist()}")
