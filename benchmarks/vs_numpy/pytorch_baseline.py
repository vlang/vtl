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


class MLP(nn.Module):
    """Three-layer f64 MLP matching the VTL benchmark's dtype and layer sizes."""

    def __init__(self):
        super().__init__()
        self.first = nn.Linear(128, 64)
        self.second = nn.Linear(64, 64)
        self.third = nn.Linear(64, 32)
        self.double()

    def forward(self, x):
        x = torch.relu(self.first(x))
        x = torch.relu(self.second(x))
        return self.third(x)


def bench_autograd():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    for batch in (32, 64):
        model = MLP()
        x = torch.ones(batch, 128, dtype=torch.float64)
        y = torch.zeros(batch, 32, dtype=torch.float64)
        criterion = nn.MSELoss()

        def step():
            model.zero_grad(set_to_none=False)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()

        for _ in range(5):
            step()
        sec = timeit.timeit(step, number=10) / 10.0
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
