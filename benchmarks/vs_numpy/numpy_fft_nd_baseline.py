"""NumPy batched-axis and N-D real FFT baselines matched to fft_nd_bench.v."""

from time import perf_counter_ns

import numpy as np


SAMPLE_COUNT = 7
WARMUP_RUNS = 3
CASES = (
    ("batched_rfft_axis_f64", (64, 512), 1, 5, np.float64, False),
    ("batched_rfft_axis_f32", (64, 512), 1, 5, np.float32, False),
    ("rfftn_3d_f64", (32, 32, 64), None, 3, np.float64, False),
    ("rfftn_3d_f32", (32, 32, 64), None, 3, np.float32, False),
    ("batched_rfft_axis_f64_stride2", (64, 512), 1, 5, np.float64, True),
    ("batched_rfft_axis_f32_stride2", (64, 512), 1, 5, np.float32, True),
    ("rfftn_3d_f64_stride2", (32, 32, 64), None, 3, np.float64, True),
    ("rfftn_3d_f32_stride2", (32, 32, 64), None, 3, np.float32, True),
)


def make_input(shape: tuple[int, ...], dtype: type[np.floating]) -> np.ndarray:
    positions = np.arange(np.prod(shape), dtype=np.float64)
    values = np.sin(positions * 0.0017) + 0.25 * np.cos(positions * 0.0023)
    return values.astype(dtype).reshape(shape)


def transform(source: np.ndarray, axis: int | None) -> np.ndarray:
    if axis is None:
        return np.fft.rfftn(source)
    return np.fft.rfft(source, axis=axis)


def inverse(spectrum: np.ndarray, shape: tuple[int, ...], axis: int | None) -> np.ndarray:
    if axis is None:
        return np.fft.irfftn(spectrum, s=shape, axes=tuple(range(len(shape))))
    return np.fft.irfft(spectrum, n=shape[axis], axis=axis)


def benchmark(
    source: np.ndarray,
    axis: int | None,
    iterations: int,
) -> tuple[float, float, float, float, float]:
    for _ in range(WARMUP_RUNS):
        transform(source, axis)
    timings: list[float] = []
    result = transform(source, axis)
    for _ in range(SAMPLE_COUNT):
        started = perf_counter_ns()
        for _ in range(iterations):
            result = transform(source, axis)
        timings.append((perf_counter_ns() - started) / iterations / 1000.0)
    timings.sort()
    restored = inverse(result, source.shape, axis)
    max_abs_error = float(np.max(np.abs(restored - source)))
    weights = np.arange(1, result.size + 1, dtype=np.float64).reshape(result.shape)
    checksum = float(np.sum(weights * (result.real + 0.5 * result.imag)))
    return timings[SAMPLE_COUNT // 2], timings[0], timings[-1], max_abs_error, checksum


def main() -> None:
    print("transform,shape,iterations,samples,median_us,min_us,max_us,max_abs_error,checksum")
    for name, shape, axis, iterations, dtype, strided in CASES:
        source = make_input(shape, dtype)
        if strided:
            base_shape = (*shape[:-1], shape[-1] * 2)
            base = np.zeros(base_shape, dtype=dtype)
            base[..., ::2] = source
            source = base[..., ::2]
        median, minimum, maximum, error, checksum = benchmark(source, axis, iterations)
        print(
            f"{name},{'x'.join(map(str, shape))},{iterations},{SAMPLE_COUNT},"
            f"{median:.3f},{minimum:.3f},{maximum:.3f},{error:.6e},{checksum:.6f}"
        )


if __name__ == "__main__":
    main()
