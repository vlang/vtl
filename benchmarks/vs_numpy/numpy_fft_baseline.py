"""NumPy real FFT baseline matched to ``fft_bench.v``."""

from time import perf_counter_ns

import numpy as np


SIZES = (256, 1024, 4096, 16384)
ITERATIONS = {256: 100, 1024: 50, 4096: 20, 16384: 10}
WARMUP_RUNS = 3
SAMPLE_COUNT = 7


def signal(size: int, dtype: type[np.floating]) -> np.ndarray:
    positions = np.arange(size, dtype=np.float64) / size
    values = np.sin(2.0 * np.pi * 7.0 * positions) + 0.25 * np.cos(
        2.0 * np.pi * 31.0 * positions
    )
    return values.astype(dtype)


def expected_spectrum(size: int) -> np.ndarray:
    expected = np.zeros(size // 2 + 1, dtype=np.complex128)
    expected[7] = -0.5j * size
    expected[31] = size / 8.0
    return expected


def benchmark(source: np.ndarray) -> tuple[float, float, float, float, float]:
    for _ in range(WARMUP_RUNS):
        np.fft.rfft(source)
    iterations = ITERATIONS[len(source)]
    durations: list[float] = []
    checksum = 0.0
    max_abs_error = 0.0
    weights = np.arange(1, len(source) // 2 + 2, dtype=np.float64)
    expected = expected_spectrum(len(source))
    for _ in range(SAMPLE_COUNT):
        started = perf_counter_ns()
        for _ in range(iterations):
            result = np.fft.rfft(source)
        durations.append((perf_counter_ns() - started) / iterations / 1000.0)
        checksum = float(np.sum(weights * (result.real + 0.5 * result.imag)))
        max_abs_error = float(np.max(np.abs(result - expected)))
    durations.sort()
    return durations[SAMPLE_COUNT // 2], durations[0], durations[-1], max_abs_error, checksum


def main() -> None:
    for label, dtype in (("f64", np.float64), ("f32 input", np.float32)):
        print(f"NumPy real {label} rfft forward transform")
        print("layout,size,iterations,samples,median_us,min_us,max_us,max_abs_error,checksum")
        for layout in ("contiguous", "strided"):
            for size in SIZES:
                source = signal(size, dtype)
                if layout == "strided":
                    interleaved = np.zeros(size * 2, dtype=dtype)
                    interleaved[::2] = source
                    source = interleaved[::2]
                median_us, min_us, max_us, max_abs_error, checksum = benchmark(source)
                print(
                    f"{layout},{size},{ITERATIONS[size]},{SAMPLE_COUNT},{median_us:.3f},"
                    f"{min_us:.3f},{max_us:.3f},{max_abs_error:.6e},{checksum:.6f}"
                )


if __name__ == "__main__":
    main()
