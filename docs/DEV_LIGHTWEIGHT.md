# Lightweight development (avoid OOM / machine freeze)

VTL/VSL test suites compile **many** modules in parallel. On a 32-core machine, `v test vsl`
or `v test vtl` can spike **RAM into the 10–20+ GB** range during compilation alone.

## Rules of thumb

| Do | Don't |
|----|--------|
| `vscoped v test ./vtl/nn/layers/layers_test.v` | `v test vtl` (full tree) |
| `vscoped v test ./vsl/blas vsl/la vsl/ml` | `v test vsl` (82 tests + Vulkan) |
| `vscoped v run ./vtl/examples/nn_cifar10_tiny_synth/main.v` | `v run ./vtl/examples/nn_cifar10/main.v` (huge compile) |
| Opt-in GPU: `vscoped env VTL_USE_CUDA=1 v -d cuda ...` | `-d cuda` on every command by default |

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
| `VTL_NO_PARALLEL` | off | Set to `1` to pass `-no-parallel` to V for each local `bin/test` command. |
| `VJOBS` | `2` locally | Cap compiler parallelism; use `1` for GPU tests when needed. |

CUDA is **opt-in** so normal CPU work never touches the GPU driver.

## Recommended commands

```bash
cd ~/.vmodules

# Keep every local V process inside the same memory ceiling. Override VJOBS for
# the few GPU probes that need a single compiler job.
vscoped() {
  systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 -- \
    env VJOBS="${VJOBS:-2}" "$@"
}

vscoped v up

# CPU smoke (fast)
vscoped v test ./vtl/nn/layers/layers_test.v
vscoped v test ./vtl/nn/models/serialization_test.v
vscoped v run ./vtl/examples/nn_cifar10_tiny_synth/main.v
vscoped v run ./vtl/examples/nn_cifar10_f32_tiny_synth/main.v
VJOBS=1 vscoped v test ./vtl/nn/f32_training_smoke_test.v ./vtl/nn/f32_autograd_smoke_test.v
VJOBS=1 vscoped env VTL_USE_CUDA=1 VTL_TEST_CUDA=1 v -d cuda test ./vtl/nn/cuda_training_smoke_test.v
VJOBS=1 vscoped env VTL_USE_CUDA=1 v -d cuda run ./vtl/examples/nn_cifar10_cuda/main.v
vscoped v run ./vtl/examples/nn_cifar10_f32_vulkan_tiny_synth/main.v
# Optional Vulkan training smoke (uncomment both lines):
# VJOBS=1 vscoped env VTL_USE_VULKAN=1 VTL_TEST_VULKAN=1 \
#   v -prod -d vulkan test ./vtl/nn/f32_vulkan_training_smoke_d_vulkan_test.v

# Single-file CUDA test (only when you want GPU)
VJOBS=1 vscoped env VTL_USE_CUDA=1 VTL_TEST_CUDA=1 v -d cuda test ./vtl/nn/layers/linear_cuda_test.v
vscoped v test ./vtl/autograd/device_session_test.v
# GPU backward parity (optional)
# Optional CUDA backward parity (uncomment both lines):
# VJOBS=1 vscoped env VTL_USE_CUDA=1 VTL_TEST_CUDA=1 VTL_CUDA_BACKWARD=1 \
#   v -d cuda test ./vtl/autograd/device_session_test.v

# VSL CUDA ops (one file)
VJOBS=1 vscoped v -d cuda test ./vsl/cuda/examples/cuda_ops_test.v

# Vulkan f32 (CPU compile path without SDK)
vscoped v test ./vtl/nn/layers/linear_vulkan_integration_test.v
vscoped v run ./vtl/examples/nn_cifar10_vulkan/main.v

# Vulkan f32 GPU full stack (Linear + Conv2D + ReLU + Adam)
VJOBS=1 vscoped env VTL_USE_VULKAN=1 v -prod -d vulkan run ./vtl/examples/nn_cifar10_vulkan/main.v
VJOBS=1 vscoped env VTL_USE_VULKAN=1 v -prod -d vulkan test ./vtl/nn/f32_vulkan_training_smoke_d_vulkan_test.v
VJOBS=1 vscoped env VTL_USE_VULKAN=1 v -prod -d vulkan test \
  ./vtl/nn/internal/conv2d_vulkan_forward_f32_d_vulkan_test.v \
  ./vtl/nn/internal/conv2d_vulkan_backward_f32_d_vulkan_test.v \
  ./vtl/nn/layers/activation_vulkan_relu_f32_d_vulkan_test.v \
  ./vtl/nn/optimizers/adam_f32_vulkan_d_vulkan_test.v
VJOBS=1 vscoped env VSL_TEST_VULKAN=1 v -prod -d vulkan test ./vsl/vulkan/compute/adam_step_vulkan_test.v
```

## CI split (implemented, #109)

- **PR default (`ci.yml`):** `bin/test` — discover every directory containing
  `*_test.v` and run each package separately to release compiler memory. Root
  tests and the large `nn`, `nn/layers`, and `nn/models` packages run one file
  at a time. The
  script also compiles examples except `nn_cifar10/main.v` and runs tiny
  synthetic CIFAR examples.
- **Label `full-ml`:** workflow `ci-full-ml.yml` — `bin/test --full`
  (weekly schedule + manual dispatch)
- **Local full scoped suite:** from `~/.vmodules`, run:

  ```bash
  systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 -- \
    env VJOBS=2 VTL_NO_PARALLEL=1 ./vtl/bin/test
  ```

  This serializes compiler transformations and releases compiler memory
  between test packages. CI does not set `VTL_NO_PARALLEL` and runs directly
  on hosted runners.
