"""Matched complex64 FFT baseline for fft_complex_f32_bench.v."""

import math
import time

import numpy as np


SIZES = (256, 1024, 4096, 16384)


def benchmark(size: int, iterations: int) -> tuple[float, float]:
    positions = np.arange(size, dtype=np.float64) / size
    values = np.empty(size, dtype=np.complex64)
    values.real = np.sin(2.0 * math.pi * 7.0 * positions).astype(np.float32)
    values.imag = (0.25 * np.cos(2.0 * math.pi * 31.0 * positions)).astype(np.float32)
    for _ in range(3):
        np.fft.fft(values)
    checksum = 0.0
    started = time.perf_counter_ns()
    for _ in range(iterations):
        checksum += float(np.fft.fft(values)[0].real)
    elapsed_ns = time.perf_counter_ns() - started
    return elapsed_ns / iterations / 1000.0, checksum


def main() -> None:
    print("NumPy complex64 FFT forward transform (NumPy promotes to complex128)")
    print("size,iterations,mean_us,checksum")
    for size in SIZES:
        iterations = 50 if size <= 1024 else 20 if size <= 4096 else 10
        mean_us, checksum = benchmark(size, iterations)
        print(f"{size},{iterations},{mean_us:.3f},{checksum:.6f}")


if __name__ == "__main__":
    main()
