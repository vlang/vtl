#!/usr/bin/env python3
"""NumPy reference for the matrix-chain ordering benchmark."""

import time

import numpy as np


def benchmark(name: str, operation, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[float, float]:
    for _ in range(2):
        operation(a, b, c)

    samples_ms = []
    checksum = 0.0
    for _ in range(10):
        started = time.perf_counter_ns()
        result = operation(a, b, c)
        samples_ms.append((time.perf_counter_ns() - started) / 1_000_000)
        checksum += float(np.sum(result))
    return sum(samples_ms) / len(samples_ms), checksum


def main() -> None:
    a = np.ones((400, 40), dtype=np.float64)
    b = np.ones((40, 4000), dtype=np.float64)
    c = np.ones((4000, 40), dtype=np.float64)
    operations = (
        ("multi_dot", lambda x, y, z: np.linalg.multi_dot([x, y, z])),
        ("left-associated", lambda x, y, z: (x @ y) @ z),
    )
    print("NumPy matrix-chain benchmark (400x40 · 40x4000 · 4000x40)")
    print("Ordering | Mean ms | Checksum")
    print("---------+---------+---------")
    for name, operation in operations:
        mean_ms, checksum = benchmark(name, operation, a, b, c)
        print(f"{name} | {mean_ms:.4f} | {checksum:.1f}")

    expected = np.full((400, 40), 160000.0)
    assert np.array_equal(np.linalg.multi_dot([a, b, c]), expected)


if __name__ == "__main__":
    main()
