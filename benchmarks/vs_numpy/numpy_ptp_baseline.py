import os
import time

import numpy as np


def main() -> None:
    size = int(os.environ.get("VTL_PTP_BENCH_SIZE", "1048576"))
    if size <= 0:
        raise SystemExit("VTL_PTP_BENCH_SIZE must be a positive integer")
    iterations = 7
    indices = np.arange(size, dtype=np.int64)
    values = (indices % 2001).astype(np.float64) - 1000.0

    for _ in range(2):
        np.ptp(values)

    samples = []
    result = 0.0
    for _ in range(iterations):
        started = time.perf_counter()
        result = np.ptp(values)
        samples.append((time.perf_counter() - started) * 1000.0)

    print(f"numpy ptp {size} f64 | {sum(samples) / iterations:.3f} ms")
    print(f"checksum: {result:.1f}")


if __name__ == "__main__":
    main()
