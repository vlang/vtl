#!/usr/bin/env python3
"""NumPy baseline matching vector_norm_f32_bench.v input and operation."""

import time

import numpy as np


def main() -> None:
    values = np.arange(500_000, dtype=np.float32) % np.float32(97)
    values = values / np.float32(13) - np.float32(3)
    for _ in range(3):
        np.linalg.vector_norm(values, ord=2, axis=0)

    samples = []
    checksum = 0.0
    for _ in range(7):
        started = time.perf_counter_ns()
        result = np.linalg.vector_norm(values, ord=2, axis=0)
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
        checksum += float(result)

    print(f"numpy vector_norm 500000 f32 | {sum(samples) / len(samples):.3f} ms")
    print(f"checksum: {checksum:.12f}")


if __name__ == "__main__":
    main()
