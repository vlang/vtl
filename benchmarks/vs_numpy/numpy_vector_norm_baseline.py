#!/usr/bin/env python3
"""NumPy baseline matching vector_norm_bench.v input and timed operation."""

import time

import numpy as np


def main() -> None:
    values = np.arange(500_000, dtype=np.float64) % 97 / 13.0 - 3.0
    for _ in range(3):
        np.linalg.vector_norm(values, ord=2, axis=0)

    samples = []
    checksum = 0.0
    for _ in range(7):
        started = time.perf_counter_ns()
        result = np.linalg.vector_norm(values, ord=2, axis=0)
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
        checksum += float(result)

    print(f"numpy vector_norm 500000 f64 | {sum(samples) / len(samples):.3f} ms")
    print(f"checksum: {checksum:.12f}")


if __name__ == "__main__":
    main()
