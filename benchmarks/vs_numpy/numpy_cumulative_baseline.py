#!/usr/bin/env python3
"""NumPy baselines for VTL cumulative sum and product benchmarks."""

import time

import numpy as np

SIZE = 1_048_576
ITERATIONS = 7


def bench(name, operation, values):
    for _ in range(2):
        operation(values)

    samples = []
    result = None
    for _ in range(ITERATIONS):
        started = time.perf_counter_ns()
        result = operation(values)
        samples.append((time.perf_counter_ns() - started) / 1_000_000)

    print(
        f"numpy {name} {SIZE} f64 | {sum(samples) / len(samples):.3f} ms "
        f"| checksum {float(np.sum(result, dtype=np.float64)):.6f}"
    )


def main():
    indices = np.arange(SIZE, dtype=np.int64)
    sum_values = (indices % 1000).astype(np.float64) / 1000.0
    product_values = 1.0 - (indices % 16).astype(np.float64) * 1e-8
    bench("cumsum", lambda values: np.cumsum(values, dtype=np.float64), sum_values)
    bench("cumprod", lambda values: np.cumprod(values, dtype=np.float64), product_values)


if __name__ == "__main__":
    main()
