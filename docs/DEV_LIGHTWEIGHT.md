# Lightweight development (avoid OOM / machine freeze)

VTL/VSL test suites compile **many** modules in parallel. On a 32-core machine, `v test vsl`
or `v test vtl` can spike **RAM into the 10–20+ GB** range during compilation alone.

## Rules of thumb

| Do | Don't |
|----|--------|
| `VJOBS=2 v test vtl/nn/layers/layers_test.v` | `v test vtl` (full tree) |
| `VJOBS=2 v test vsl/blas vsl/la vsl/ml` | `v test vsl` (82 tests + Vulkan) |
| `v run vtl/examples/nn_cifar10_tiny_synth/main.v` | `v run vtl/examples/nn_cifar10/main.v` (huge compile) |
| Opt-in GPU: `VTL_USE_CUDA=1 v -d cuda ...` | `-d cuda` on every command by default |

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `VTL_USE_CUDA` | off | Set to `1` to use CUDA in Linear forward (`-d cuda` build only). |
| `VTL_GPU_ACTIVATIONS` | off | Phase 2: chain Linear activations on GPU between layers. |
| `VTL_CUDA_BACKWARD` | off | Phase 3: cuBLAS GEMM for Linear gate backward. |
| `VTL_TEST_CUDA` | off | Set to `1` to run GPU tests (`linear_cuda_test.v`, `device_session_test.v`). |
| `VJOBS` | (V auto) | Cap parallel compile jobs, e.g. `VJOBS=2`. |

CUDA is **opt-in** so normal CPU work never touches the GPU driver.

## Recommended commands

```bash
v up
cd ~/.vmodules

# CPU smoke (fast)
VJOBS=2 v test vtl/nn/layers/layers_test.v
VJOBS=2 v test vtl/nn/models/serialization_test.v
v run vtl/examples/nn_cifar10_tiny_synth/main.v

# Single-file CUDA test (only when you want GPU)
VTL_USE_CUDA=1 VTL_TEST_CUDA=1 VJOBS=1 v -d cuda test vtl/nn/layers/linear_cuda_test.v
VJOBS=2 v test vtl/autograd/device_session_test.v
# GPU backward parity (optional)
# VTL_USE_CUDA=1 VTL_TEST_CUDA=1 VTL_CUDA_BACKWARD=1 VJOBS=1 v -d cuda test vtl/autograd/device_session_test.v

# VSL CUDA ops (one file)
VJOBS=1 v -d cuda test vsl/cuda/examples/cuda_ops_test.v
```

## CI split (suggested)

- **PR default:** CPU scoped tests + `nn_cifar10_tiny_synth`
- **Label `cuda`:** optional workflow with `VTL_USE_CUDA=1 -d cuda` on GPU runner
- **Nightly:** full matrix with low parallelism
