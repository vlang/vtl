import time

import numpy as np


def main() -> None:
    size = 1_048_576
    iterations = 7
    indices = np.arange(size, dtype=np.int64)
    left = ((indices % 1000) - 500).astype(np.float64) * 0.25
    right = ((indices % 997) - 498).astype(np.float64) * 0.25

    for _ in range(2):
        np.maximum(left, right)
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        result = np.maximum(left, right)
        samples.append((time.perf_counter() - started) * 1000.0)

    print(f"numpy maximum {size} f64 | {sum(samples) / iterations:.3f} ms")
    checksum = int(np.sum(result * 4, dtype=np.int64))
    print(f"checksum: {checksum}")


if __name__ == "__main__":
    main()
