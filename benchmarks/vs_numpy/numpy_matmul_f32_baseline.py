#!/usr/bin/env python3
"""NumPy f32 GEMM baseline for the resident-buffer Vulkan benchmark."""

import time

import numpy as np


def bench_matmul() -> None:
    for n in (128, 256, 512, 1024, 2048):
        i, j = np.indices((n, n), dtype=np.int64)
        a = ((((i * 17 + j * 13) % 997) + 1).astype(np.float32)) / np.float32(997)
        b = ((((i * 7 + j * 19) % 991) + 1).astype(np.float32)) / np.float32(991)
        output = np.empty((n, n), dtype=np.float32)
        for _ in range(3):
            np.matmul(a, b, out=output)
        samples = []
        for _ in range(10):
            started = time.perf_counter_ns()
            np.matmul(a, b, out=output)
            samples.append((time.perf_counter_ns() - started) / 1_000_000)
        checksum = float(output[n // 2, n // 2])
        mean_ms = sum(samples) / len(samples)
        print(f"{n},10,{mean_ms:.3f},{checksum:.6f}")


if __name__ == "__main__":
    bench_matmul()
