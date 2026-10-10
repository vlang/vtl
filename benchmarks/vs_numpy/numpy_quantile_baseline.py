#!/usr/bin/env python3
"""NumPy reference for the VTL scalar median selection benchmark."""

import time

import numpy as np


def bench_quantile() -> None:
    for n in (100_000, 1_000_000):
        values = ((np.arange(n, dtype=np.int64) * 7919 + 17) % 1009) / 1009.0
        for _ in range(2):
            np.quantile(values, 0.5, method="linear")
        samples = []
        checksum = 0.0
        for _ in range(7):
            started = time.perf_counter_ns()
            checksum += float(np.quantile(values, 0.5, method="linear"))
            samples.append((time.perf_counter_ns() - started) / 1_000_000)
        print(f"numpy_quantile | {n} | {sum(samples) / len(samples):.4f} | {checksum:.7f}")


if __name__ == "__main__":
    bench_quantile()
