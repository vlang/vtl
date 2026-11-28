#!/usr/bin/env python3
"""PyTorch baselines matched to pytorch_pair_bench.v.

The workloads, dtypes, warmup, repetitions, and checksums mirror the VTL
benchmark so the two sets of rows can be paired directly.
"""

import os
import statistics
import sys
import time

try:
    import torch
except ImportError:
    print("PyTorch not installed - skip", file=sys.stderr)
    sys.exit(0)

WARMUP = 3
ITERATIONS = 10


def time_op(operation):
    """Return (samples_ms, last_result); sync device work before recording."""
    samples = []
    result = None
    for _ in range(ITERATIONS):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter_ns()
        result = operation()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
    return samples, result


def report(op, size, samples, checksum):
    median = statistics.median(samples)
    spread = statistics.stdev(samples) if len(samples) > 1 else 0.0
    print(f"pytorch | {op} | {size} | {median:.4f} | {spread:.4f} | {checksum:.4f}")


def vector(n):
    return (torch.arange(n, device=device, dtype=torch.float64) % 1000 + 1) / 1000.0


def bench_stat(op, n):
    values = vector(n)

    def operation():
        return torch.sum(values) if op == "sum" else torch.mean(values)

    for _ in range(WARMUP):
        operation()
    samples, _ = time_op(operation)
    checksum = 0.0
    for _ in range(ITERATIONS):
        checksum += float(operation())
    report(op, n, samples, checksum)


def bench_tanh(n):
    values = vector(n)

    def operation():
        return torch.tanh(values)

    for _ in range(WARMUP):
        operation()
    samples, _ = time_op(operation)
    checksum = 0.0
    for _ in range(ITERATIONS):
        checksum += float(operation()[n // 2])
    report("tanh", n, samples, checksum)


def bench_gemm(n):
    i = torch.arange(n, device=device).view(n, 1)
    j = torch.arange(n, device=device).view(1, n)
    a = (((i * 17 + j * 13) % 997) + 1).double() / 997.0
    b = (((i * 7 + j * 19) % 991) + 1).double() / 991.0

    def operation():
        return a @ b

    for _ in range(WARMUP):
        operation()
    samples, _ = time_op(operation)
    checksum = 0.0
    for _ in range(ITERATIONS):
        checksum += float(operation()[n // 2, n // 2])
    report("gemm", f"{n}x{n}", samples, checksum)


def main():
    global device
    name = os.getenv("PYTORCH_BENCH_DEVICE", "cpu")
    if name.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(name)
    else:
        device = torch.device("cpu")
    torch.set_num_threads(2)
    print(
        f"pytorch | version={torch.__version__} | device={device.type} "
        f"| dtype=float64 | threads={torch.get_num_threads()} "
        f"| MKL={torch.backends.mkl.is_available()} "
        f"| oneDNN={torch.backends.mkldnn.is_available() and torch.backends.mkldnn.enabled} "
        f"| warmup={WARMUP} | iterations={ITERATIONS}"
    )
    command = sys.argv[1] if len(sys.argv) > 1 else "all"
    workloads = {
        "sum": lambda: [bench_stat("sum", n) for n in (100_000, 1_000_000)],
        "mean": lambda: [bench_stat("mean", n) for n in (100_000, 1_000_000)],
        "tanh": lambda: [bench_tanh(n) for n in (100_000, 1_000_000)],
        "gemm": lambda: [bench_gemm(n) for n in (128, 256, 512)],
    }
    if command == "all":
        for run in workloads.values():
            run()
    elif command in workloads:
        workloads[command]()
    else:
        print(f"unknown: {command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
