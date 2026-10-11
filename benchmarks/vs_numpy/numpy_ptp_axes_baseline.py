import os
import time

import numpy as np


def main() -> None:
    groups = int(os.environ.get("VTL_PTP_AXES_BENCH_GROUPS", "128"))
    if groups <= 0:
        raise SystemExit("VTL_PTP_AXES_BENCH_GROUPS must be a positive integer")
    size = groups * 256 * 512
    values = ((np.arange(size, dtype=np.int32) % 2001) - 1000).astype(np.float64)
    values = values.reshape(groups, 256, 512)
    values *= (np.arange(groups, dtype=np.float64) % 8 + 1)[:, None, None]
    iterations = 5

    for _ in range(2):
        np.ptp(values, axis=(1, 2))

    samples = []
    result = None
    for _ in range(iterations):
        started = time.perf_counter()
        result = np.ptp(values, axis=(1, 2))
        samples.append((time.perf_counter() - started) * 1000.0)

    print(f"numpy ptp axes {size} f64 | {sum(samples) / iterations:.3f} ms")
    print(f"checksum: {float(result.sum()):.1f}")


if __name__ == "__main__":
    main()
