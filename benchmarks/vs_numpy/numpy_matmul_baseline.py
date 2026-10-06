#!/usr/bin/env python3
"""NumPy baseline matching the end-to-end VTL matmul benchmark."""

import time

import numpy as np


def bench_matmul() -> None:
    for n in (128, 256, 512):
        i, j = np.indices((n, n), dtype=np.int64)
        a = (((i * 17 + j * 13) % 997) + 1).astype(np.float64) / 997.0
        b = (((i * 7 + j * 19) % 991) + 1).astype(np.float64) / 991.0
        for _ in range(3):
            result = a @ b
        samples = []
        checksum = 0.0
        for _ in range(10):
            started = time.perf_counter_ns()
            result = a @ b
            samples.append((time.perf_counter_ns() - started) / 1_000_000)
            checksum += result[n // 2, n // 2]
        mean_ms = sum(samples) / len(samples)
        gflops = 2 * n**3 / (mean_ms / 1000) / 1e9
        print(f"numpy gemm {n}x{n} | {mean_ms:.3f} ms | {gflops:.3f} GFLOPS | {checksum:.6f}")


if __name__ == "__main__":
    bench_matmul()
