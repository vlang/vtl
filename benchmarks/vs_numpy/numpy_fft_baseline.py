"""NumPy real FFT baseline matched to ``fft_bench.v``."""

from time import perf_counter_ns

import numpy as np


SIZES = (256, 1024, 4096, 16384)
ITERATIONS = {256: 100, 1024: 50, 4096: 20, 16384: 10}
WARMUP_RUNS = 3


def signal(size: int, dtype: type[np.floating]) -> np.ndarray:
    positions = np.arange(size, dtype=np.float64) / size
    values = np.sin(2.0 * np.pi * 7.0 * positions) + 0.25 * np.cos(
        2.0 * np.pi * 31.0 * positions
    )
    return values.astype(dtype)


def benchmark(source: np.ndarray) -> tuple[float, float]:
    for _ in range(WARMUP_RUNS):
        np.fft.rfft(source)
    checksum = 0.0
    iterations = ITERATIONS[len(source)]
    started = perf_counter_ns()
    for _ in range(iterations):
        checksum += float(np.fft.rfft(source)[0].real)
    elapsed_ns = perf_counter_ns() - started
    return elapsed_ns / iterations / 1000.0, checksum


def main() -> None:
    for label, dtype in (("f64", np.float64), ("f32 input", np.float32)):
        print(f"NumPy real {label} rfft forward transform")
        print("size,iterations,mean_us,checksum")
        for size in SIZES:
            mean_us, checksum = benchmark(signal(size, dtype))
            print(f"{size},{ITERATIONS[size]},{mean_us:.3f},{checksum:.6f}")


if __name__ == "__main__":
    main()
