import time

import numpy as np


def benchmark(dtype: type[np.floating], dtype_name: str) -> None:
    size = 1_048_576
    iterations = 7
    indices = np.arange(size, dtype=np.int64)
    left = ((indices % 1000) - 500).astype(dtype) * dtype(0.25)
    right = ((indices % 997) - 498).astype(dtype) * dtype(0.25)

    for _ in range(2):
        np.maximum(left, right)
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        result = np.maximum(left, right)
        samples.append((time.perf_counter() - started) * 1000.0)

    print(f"numpy maximum {size} {dtype_name} | {sum(samples) / iterations:.3f} ms")
    checksum = int(np.sum(result * dtype(4), dtype=np.int64))
    print(f"checksum: {checksum}")


def main() -> None:
    benchmark(np.float32, "f32")
    benchmark(np.float64, "f64")


if __name__ == "__main__":
    main()
