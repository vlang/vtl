#!/usr/bin/env python3
"""NumPy baselines for contiguous VTL unary autograd backward gates."""

import platform
import time

import numpy as np


ELEMENTS = 1_048_576
WARMUPS = 3
ITERATIONS = 9
gradient = np.full(ELEMENTS, 1.25, dtype=np.float64)
tanh_cache = np.full(ELEMENTS, 0.5, dtype=np.float64)
abs_input = np.full(ELEMENTS, 0.5, dtype=np.float64)
sqrt_input = np.full(ELEMENTS, 4.0, dtype=np.float64)
clamp_input = np.zeros(ELEMENTS, dtype=np.float64)


def backward(operation):
    if operation == "tanh":
        scratch = np.square(tanh_cache)
        np.subtract(1.0, scratch, out=scratch)
        return np.multiply(gradient, scratch)
    if operation == "abs":
        scratch = np.abs(abs_input)
        np.maximum(scratch, 1e-8, out=scratch)
        np.divide(abs_input, scratch, out=scratch)
        return np.multiply(gradient, scratch)
    if operation == "sqrt":
        scratch = np.sqrt(sqrt_input)
        np.multiply(scratch, 2.0, out=scratch)
        return np.divide(gradient, scratch)
    if operation == "clamp":
        mask = np.logical_and(clamp_input >= -1.0, clamp_input <= 1.0)
        return np.where(mask, gradient, 0.0)
    raise ValueError(f"unknown operation: {operation}")


for operation in ("tanh", "abs", "sqrt", "clamp"):
    for _ in range(WARMUPS):
        backward(operation)

    samples = []
    for _ in range(ITERATIONS):
        start = time.perf_counter_ns()
        output = backward(operation)
        samples.append((time.perf_counter_ns() - start) / 1_000_000.0)

    samples.sort()
    checksum = float(output[0] * ELEMENTS)
    print(
        f"NumPy {np.__version__} | CPU={platform.processor() or platform.machine()} | "
        f"op={operation} elements={ELEMENTS} "
        f"median_ms={samples[len(samples) // 2]:.3f} checksum={checksum:.3f}"
    )
