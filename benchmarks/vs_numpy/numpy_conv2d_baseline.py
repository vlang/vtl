#!/usr/bin/env python3
"""NumPy Conv2D baseline matching conv2d_bench.v."""

import time

import numpy as np

BATCH = 1
IN_CHANNELS = 4
OUT_CHANNELS = 8
HEIGHT = 32
WIDTH = 32
KERNEL = 3
WARMUPS = 3
ITERATIONS = 20


def inputs(dtype: type[np.floating]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_indices = np.arange(BATCH * IN_CHANNELS * HEIGHT * WIDTH, dtype=np.int64)
    weight_indices = np.arange(OUT_CHANNELS * IN_CHANNELS * KERNEL * KERNEL, dtype=np.int64)
    x = ((input_indices * 13) % 101 - 50).astype(dtype) / 101.0
    weights = ((weight_indices * 7) % 37 - 18).astype(dtype) / 37.0
    bias = np.arange(1, OUT_CHANNELS + 1, dtype=dtype) / 10.0
    return (
        x.reshape(BATCH, IN_CHANNELS, HEIGHT, WIDTH),
        weights.reshape(OUT_CHANNELS, IN_CHANNELS, KERNEL, KERNEL),
        bias,
    )


def conv2d(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (KERNEL, KERNEL), axis=(2, 3))
    result = np.einsum("nchwkl,ockl->nohw", windows, weights, optimize=True)
    return result + bias.reshape(1, OUT_CHANNELS, 1, 1)


def benchmark(dtype: type[np.floating]) -> None:
    x, weights, bias = inputs(dtype)
    for _ in range(WARMUPS):
        output = conv2d(x, weights, bias)

    samples_ns = []
    for _ in range(ITERATIONS):
        started = time.perf_counter_ns()
        output = conv2d(x, weights, bias)
        samples_ns.append(time.perf_counter_ns() - started)

    mean_ms = float(np.mean(samples_ns)) / 1_000_000.0
    checksum = float(np.sum(output, dtype=np.float64))
    print(f"NumPy {np.__version__}; {np.dtype(dtype)}; warmups={WARMUPS}; iterations={ITERATIONS}")
    print(
        f"conv2d {np.dtype(dtype)} | 1x4x32x32, 8x4x3x3 | {mean_ms:.6f} | {checksum:.12f}"
    )


def main() -> None:
    print("Benchmark | Size | Avg (ms) | Checksum")
    benchmark(np.float64)
    benchmark(np.float32)


if __name__ == "__main__":
    main()
