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
| `VTL_CUDA_OPTIMIZER` | off | Phase 4: cuBLAS moment updates for Adam. |
| `VTL_TEST_CUDA` | off | Set to `1` to run GPU tests (`linear_cuda_test.v`, `device_session_test.v`). |
| `VTL_USE_VULKAN` | off | f32 Linear/Conv2D/activations/Adam via Vulkan (`-d vulkan`). Use `v -prod` for GPU (debug instance crash on V 0.5.1). |
| `VTL_TEST_VULKAN` | off | Optional Vulkan integration tests (conv2d, activations, Adam, training smoke). |
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
VJOBS=2 v run vtl/examples/nn_cifar10_f32_tiny_synth/main.v
VJOBS=1 v test vtl/nn/f32_training_smoke_test.v vtl/nn/f32_autograd_smoke_test.v
VTL_USE_CUDA=1 VTL_TEST_CUDA=1 VJOBS=1 v -d cuda test vtl/nn/cuda_training_smoke_test.v
VTL_USE_CUDA=1 VJOBS=1 v -d cuda run vtl/examples/nn_cifar10_cuda/main.v
VJOBS=2 v run vtl/examples/nn_cifar10_f32_vulkan_tiny_synth/main.v
# VTL_USE_VULKAN=1 VTL_TEST_VULKAN=1 VJOBS=1 v -d vulkan test vtl/nn/f32_vulkan_training_smoke_test.v

# Single-file CUDA test (only when you want GPU)
VTL_USE_CUDA=1 VTL_TEST_CUDA=1 VJOBS=1 v -d cuda test vtl/nn/layers/linear_cuda_test.v
VJOBS=2 v test vtl/autograd/device_session_test.v
# GPU backward parity (optional)
# VTL_USE_CUDA=1 VTL_TEST_CUDA=1 VTL_CUDA_BACKWARD=1 VJOBS=1 v -d cuda test vtl/autograd/device_session_test.v

# VSL CUDA ops (one file)
VJOBS=1 v -d cuda test vsl/cuda/examples/cuda_ops_test.v

# Vulkan f32 (CPU compile path without SDK)
VJOBS=2 v test vtl/nn/layers/linear_vulkan_integration_test.v
v run vtl/examples/nn_cifar10_vulkan/main.v

# Vulkan f32 GPU full stack (Linear + Conv2D + ReLU + Adam)
VTL_USE_VULKAN=1 VJOBS=1 v -prod -d vulkan run vtl/examples/nn_cifar10_vulkan/main.v
VTL_USE_VULKAN=1 VJOBS=1 v -prod -d vulkan test vtl/nn/f32_vulkan_training_smoke_test_d_vulkan.v
VTL_USE_VULKAN=1 VJOBS=1 v -prod -d vulkan test \
  vtl/nn/internal/conv2d_vulkan_forward_f32_d_vulkan_test.v \
  vtl/nn/internal/conv2d_vulkan_backward_f32_d_vulkan_test.v \
  vtl/nn/layers/activation_vulkan_relu_f32_d_vulkan_test.v \
  vtl/nn/optimizers/adam_f32_vulkan_d_vulkan_test.v
VSL_TEST_VULKAN=1 VJOBS=1 v -prod -d vulkan test vsl/vulkan/compute/adam_step_vulkan_test.v
```

## CI split (implemented, #109)

- **PR default (`ci.yml`):** `bin/test` — scoped `v test` (nn, autograd, datasets, …) + compile examples except `nn_cifar10/main.v` + run `nn_cifar10_tiny_synth` and `nn_cifar10_f32_tiny_synth`
- **Label `full-ml`:** workflow `ci-full-ml.yml` — `bin/test --full` (weekly schedule + manual dispatch)
- **Local full suite:** `VJOBS=1 ~/.vmodules/vtl/bin/test --full`
