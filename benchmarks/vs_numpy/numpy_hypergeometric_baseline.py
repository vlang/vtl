"""NumPy baseline for hypergeometric_bench.v."""

import time

import numpy as np


OUTPUT_SIZE = 100_000
GENERATOR = np.random.default_rng(42)

for _ in range(2):
    GENERATOR.hypergeometric(50_000, 50_000, 50_000, size=OUTPUT_SIZE)

samples_ms = []
checksum = 0
for _ in range(7):
    started = time.perf_counter_ns()
    values = GENERATOR.hypergeometric(50_000, 50_000, 50_000, size=OUTPUT_SIZE)
    samples_ms.append((time.perf_counter_ns() - started) / 1_000_000)
    checksum += int(values[0])

print(f"NumPy hypergeometric mean: {sum(samples_ms) / len(samples_ms):.3f} ms")
print(f"samples: {samples_ms}")
print(f"checksum: {checksum}")
