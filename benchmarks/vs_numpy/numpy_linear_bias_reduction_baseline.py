#!/usr/bin/env python3
"""NumPy baseline for the VTL linear bias reduction microbenchmark."""

import time

import numpy as np


SHAPES = (
    (32, 64),
    (64, 64),
    (64, 256),
    (128, 128),
    (256, 64),
    (128, 256),
    (1024, 256),
    (4096, 512),
)
WARMUPS = 5
ITERATIONS = 50


def _benchmark(name: str, operation, matrix: np.ndarray) -> tuple[float, np.ndarray]:
    for _ in range(WARMUPS):
        result = operation(matrix)

    samples_ms = []
    for _ in range(ITERATIONS):
        started = time.perf_counter_ns()
        result = operation(matrix)
        samples_ms.append((time.perf_counter_ns() - started) / 1_000_000)

    mean_ms = sum(samples_ms) / len(samples_ms)
    print(f"{name} | {mean_ms:.4f} ms | checksum={float(result.sum()):.6f}")
    return mean_ms, result


def main() -> None:
    print(f"NumPy {np.__version__}; float32; warmups={WARMUPS}; iterations={ITERATIONS}")
    np.__config__.show()
    print("batch,features,numpy_sum_ms,numpy_matmul_ms,matmul_over_sum")

    for batch, features in SHAPES:
        count = batch * features
        values = (np.arange(count, dtype=np.int64) * 17 % 101).astype(np.float32)
        matrix = (values / np.float32(101)).reshape(batch, features)
        ones = np.ones((1, batch), dtype=np.float32)

        sum_ms, summed = _benchmark(
            "numpy_sum", lambda array: np.sum(array, axis=0, keepdims=True), matrix
        )
        matmul_ms, product = _benchmark("numpy_ones_matmul", lambda array: ones @ array, matrix)
        np.testing.assert_allclose(summed, product, rtol=1e-5, atol=1e-5)
        print(f"{batch},{features},{sum_ms:.4f},{matmul_ms:.4f},{matmul_ms / sum_ms:.3f}")


if __name__ == "__main__":
    main()
