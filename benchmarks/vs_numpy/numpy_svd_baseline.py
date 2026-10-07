"""Matched reduced SVD baseline for benchmarks/vs_numpy/svd_bench.v."""

from time import perf_counter

import numpy as np


def main() -> None:
    print("NumPy reduced SVD benchmark (float64)")
    print("Size       | Avg (ms) | Singular-value checksum")
    print("-----------+----------+-------------------------")
    for size in (16, 32, 64, 128):
        matrix = np.fromfunction(
            lambda row, column: ((row * 17 + column * 13) % 997 + 1) / 997.0,
            (size, size),
            dtype=np.float64,
        )
        for _ in range(2):
            np.linalg.svd(matrix, full_matrices=False)
        samples = []
        checksum = 0.0
        for _ in range(5):
            started = perf_counter()
            _, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
            samples.append((perf_counter() - started) * 1000.0)
            checksum += float(np.sum(singular_values))
        print(f"{size}x{size:<6} | {sum(samples) / len(samples):.6f} | {checksum:.12g}")


if __name__ == "__main__":
    main()
