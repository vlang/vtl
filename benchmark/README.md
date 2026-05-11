# VTL Backend Benchmark

This benchmark compares matrix multiplication (`matmul`) across:

- CPU (`vtl.la.matmul`)
- Vulkan backend (`-d vulkan`)
- OpenCL/VCL backend (`-d vcl`)

## What it runs

`benchmark/main.v` executes `matmul` over square matrices for sizes:

- `64 x 64`
- `128 x 128`
- `256 x 256`

Each size is repeated 3 times and averaged.

## Run commands

```sh
# CPU only
v run benchmark/main.v

# CPU + OpenCL/VCL
v -d vcl run benchmark/main.v

# CPU + Vulkan
v -d vulkan run benchmark/main.v

# CPU + both GPU backends
v -d vulkan -d vcl run benchmark/main.v
```

## Output format

Each line prints timing and speedup against CPU:

```text
matmul [128x128]: CPU=8.12ms | Vulkan=1.20ms (6.77x) | VCL=1.55ms (5.24x)
```

If a backend is not enabled, it is omitted.

## Notes

- Benchmarks are sensitive to GPU model, thermal state, and driver/runtime.
- First run can include setup overhead (shader/kernel setup, memory init).
- Use repeated runs for stable comparisons.
