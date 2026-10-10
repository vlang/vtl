#!/usr/bin/env python3
"""NumPy baseline matching promoted_remainder_bench.v."""

import time

import numpy as np


def main() -> None:
    size = 1_048_576
    values = ((np.arange(size, dtype=np.int64) % 251) - 125).astype(np.int8)
    divisor = np.array([17], dtype=np.int16)

    for _ in range(2):
        np.remainder(values, divisor)

    samples_ms = []
    result = None
    for _ in range(7):
        started = time.perf_counter_ns()
        result = np.remainder(values, divisor)
        samples_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)

    assert result is not None
    checksum = int(result.sum(dtype=np.int64))
    print(f"numpy remainder {size} i8 % i16 scalar broadcast | {sum(samples_ms) / len(samples_ms):.3f} ms")
    print(f"checksum: {checksum}")


if __name__ == "__main__":
    main()
