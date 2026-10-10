#!/usr/bin/env python3
"""NumPy baseline matching putmask_bench.v inputs and timed operation."""

import time

import numpy as np


def main() -> None:
    size = 1_000_000
    initial = np.arange(size, dtype=np.float64) % 97
    mask = np.arange(size) % 2 == 1
    updates = np.array([101.0, 202.0, 303.0], dtype=np.float64)
    for _ in range(2):
        np.putmask(initial.copy(), mask, updates)

    samples = []
    checksum = 0.0
    for _ in range(7):
        target = initial.copy()
        started = time.perf_counter_ns()
        np.putmask(target, mask, updates)
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
        checksum += float(target.sum())

    print(f"numpy putmask {size} f64, 50% mask, 3 updates | {sum(samples) / len(samples):.3f} ms")
    print(f"checksum: {checksum:.3f}")


if __name__ == "__main__":
    main()
