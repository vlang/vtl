# VTL vs NumPy baselines

Run VTL benchmarks from `~/.vmodules`, then compare with the Python snippets below.

## Matmul

```bash
v run vtl/benchmarks/vs_numpy/matmul_bench.v
```

```python
import numpy as np
import timeit
n = 1024
a = np.random.rand(n, n)
b = np.random.rand(n, n)
print(timeit.timeit(lambda: a @ b, number=10))
```

## Conv2D (CPU path)

```bash
v run vtl/benchmarks/vs_numpy/conv2d_bench.v
```

## Autograd (3-layer MLP backprop)

```bash
v run vtl/benchmarks/vs_numpy/autograd_bench.v
python3 vtl/benchmarks/vs_numpy/pytorch_baseline.py autograd
```

## Notes

- Use the same matrix sizes when comparing manually.
- Report GFLOPS from the VTL script output for PR comments.
- CUDA paths require `VTL_USE_CUDA=1` and `-d cuda` where applicable.
