import time

import numpy as np


sample_count = 262_144
bin_count = 64
repetitions = 7
indices = np.arange(sample_count, dtype=np.int64)
x = (indices * 17 % 65_536).astype(np.float64) / 65_535.0
y = ((indices * 29 + 7) % 65_536).astype(np.float64) / 65_535.0
edges = np.linspace(0.0, 1.0, bin_count + 1)

for _ in range(2):
    np.histogram2d(x, y, bins=(edges, edges))

samples = []
checksum = 0
for _ in range(repetitions):
    started = time.perf_counter_ns()
    counts, _, _ = np.histogram2d(x, y, bins=(edges, edges))
    samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    weights = np.arange(1, counts.size + 1, dtype=np.int64).reshape(counts.shape)
    checksum += int(np.sum(counts * weights))

samples.sort()
print(f"operation=histogram2d samples={sample_count} bins={bin_count}x{bin_count}")
print(f"median_ms={samples[len(samples) // 2]:.3f} weighted_checksum={checksum // repetitions}")
