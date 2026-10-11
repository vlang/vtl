import os
import time

import numpy as np


def main() -> None:
    size = int(os.environ.get("VTL_BCE_BENCH_SIZE", "524288"))
    if size <= 0:
        raise SystemExit("VTL_BCE_BENCH_SIZE must be a positive integer")
    probabilities = 0.000001 + (np.arange(size, dtype=np.float64) % 999_999) * (
        0.999998 / 999_999.0
    )
    targets = (np.arange(size) % 2 == 0).astype(np.float64)

    def calculate() -> np.float64:
        clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
        values = -targets * np.log(clipped) - (1.0 - targets) * np.log(1.0 - clipped)
        return np.mean(values)

    for _ in range(1):
        calculate()
    samples = []
    checksum = np.float64(0.0)
    for _ in range(3):
        started = time.perf_counter()
        checksum = calculate()
        samples.append((time.perf_counter() - started) * 1000.0)
    samples.sort()
    print(f"NumPy | {samples[len(samples) // 2]:.3f} ms")
    print(f"checksum: {checksum:.12f}")


if __name__ == "__main__":
    main()
