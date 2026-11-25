"""Matched complex64 FFT baseline for fft_complex_f32_bench.v."""

import math
import time

import numpy as np


SIZES = (256, 1024, 4096, 16384)
SAMPLE_COUNT = 7


def benchmark(size: int, iterations: int, strided: bool) -> tuple[float, float, float, float, float]:
    positions = np.arange(size, dtype=np.float64) / size
    values = np.empty(size, dtype=np.complex64)
    values.real = np.sin(2.0 * math.pi * 7.0 * positions).astype(np.float32)
    values.imag = (0.25 * np.cos(2.0 * math.pi * 31.0 * positions)).astype(np.float32)
    if strided:
        interleaved = np.zeros(size * 2, dtype=np.complex64)
        interleaved[::2] = values
        values = interleaved[::2]
    for _ in range(3):
        np.fft.fft(values)
    expected = np.zeros(size, dtype=np.complex128)
    expected[7] = -0.5j * size
    expected[size - 7] = 0.5j * size
    expected[31] = 0.125j * size
    expected[size - 31] = 0.125j * size
    weights = np.arange(1, size + 1, dtype=np.float64)
    durations: list[float] = []
    checksum = 0.0
    max_abs_error = 0.0
    for _ in range(SAMPLE_COUNT):
        started = time.perf_counter_ns()
        for _ in range(iterations):
            result = np.fft.fft(values)
        durations.append((time.perf_counter_ns() - started) / iterations / 1000.0)
        checksum = float(np.sum(weights * (result.real + 0.5 * result.imag)))
        max_abs_error = float(np.max(np.abs(result - expected)))
    durations.sort()
    return durations[SAMPLE_COUNT // 2], durations[0], durations[-1], max_abs_error, checksum


def main() -> None:
    print("NumPy complex64 FFT forward transform (NumPy promotes to complex128)")
    print("layout,size,iterations,samples,median_us,min_us,max_us,max_abs_error,checksum")
    for layout in ("contiguous", "strided"):
        for size in SIZES:
            iterations = 50 if size <= 1024 else 20 if size <= 4096 else 10
            median_us, min_us, max_us, max_abs_error, checksum = benchmark(
                size, iterations, layout == "strided"
            )
            print(
                f"{layout},{size},{iterations},{SAMPLE_COUNT},{median_us:.3f},"
                f"{min_us:.3f},{max_us:.3f},{max_abs_error:.6e},{checksum:.6f}"
            )


if __name__ == "__main__":
    main()
