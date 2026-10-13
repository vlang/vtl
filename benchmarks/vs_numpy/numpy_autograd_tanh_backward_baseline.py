#!/usr/bin/env python3
"""NumPy contiguous elementwise equivalent of VTL's tanh backward kernel."""

import os
import platform
import time

import numpy as np


ELEMENTS = int(os.getenv("VTL_TANH_BENCH_SIZE", "1048576"))
WARMUPS = 3
ITERATIONS = 9

gradient = np.full(ELEMENTS, 1.25, dtype=np.float64)
cached_tanh = np.full(ELEMENTS, 0.5, dtype=np.float64)
def backward():
    scratch = np.square(cached_tanh)
    np.subtract(1.0, scratch, out=scratch)
    return np.multiply(gradient, scratch)


for _ in range(WARMUPS):
    backward()

samples = []
for _ in range(ITERATIONS):
    start = time.perf_counter_ns()
    output = backward()
    samples.append((time.perf_counter_ns() - start) / 1_000_000.0)

samples.sort()
checksum = float(output[0] * ELEMENTS)
print(
    f"NumPy {np.__version__} | CPU={platform.processor() or platform.machine()} | "
    f"elements={ELEMENTS} median_ms={samples[len(samples) // 2]:.3f} "
    f"checksum={checksum:.3f}"
)
