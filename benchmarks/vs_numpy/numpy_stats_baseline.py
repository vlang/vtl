#!/usr/bin/env python3
"""NumPy reference for the VTL contiguous sum/mean benchmark."""

import time

import numpy as np


def bench_stats() -> None:
    for n in (100_000, 1_000_000):
        values = (np.arange(n, dtype=np.float64) % 1000 + 1) / 1000.0
        for name, operation in (("numpy_sum", np.sum), ("numpy_mean", np.mean)):
            for _ in range(3):
                operation(values)
            samples = []
            checksum = 0.0
            for _ in range(10):
                started = time.perf_counter_ns()
                checksum += float(operation(values))
                samples.append((time.perf_counter_ns() - started) / 1_000_000)
            mean_ms = sum(samples) / len(samples)
            print(f"{name} | {n} | {mean_ms:.4f} | checksum={checksum:.4f}")


if __name__ == "__main__":
    bench_stats()
