#!/usr/bin/env python3
"""PyTorch timing baselines for VTL vs_numpy benchmarks."""

import sys
import timeit

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("PyTorch not installed — skip", file=sys.stderr)
    sys.exit(0)


def bench_autograd():
    for batch in (64, 128):
        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        x = torch.ones(batch, 128)
        y = torch.zeros(batch, 32)
        criterion = nn.MSELoss()

        def step():
            model.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()

        sec = timeit.timeit(step, number=3) / 3.0
        print(f"pytorch mlp_backprop {batch}x128 | {sec * 1000:.2f} ms | -")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "autograd"
    if cmd == "autograd":
        bench_autograd()
    else:
        print(f"unknown: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
