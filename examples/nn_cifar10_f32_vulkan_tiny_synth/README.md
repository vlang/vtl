# nn_cifar10_f32_vulkan_tiny_synth

f32 **training** smoke (forward, MSE, backprop, Adam). **Linear** forward/backward can use Vulkan GEMM.

## Run

```bash
# CPU Linear (default)
v run vtl/examples/nn_cifar10_f32_vulkan_tiny_synth/main.v

# Vulkan GPU (requires SDK; use -prod — debug Vulkan instance crashes on V 0.5.1)
VTL_USE_VULKAN=1 v -prod -d vulkan run vtl/examples/nn_cifar10_f32_vulkan_tiny_synth/main.v
```

## Test

```bash
VJOBS=1 v test vtl/nn/f32_vulkan_training_smoke_test.v
VTL_USE_VULKAN=1 VTL_TEST_VULKAN=1 v -prod -d vulkan test vtl/nn/layers/linear_vulkan_gpu_d_vulkan_test.v
```

CPU-only f32 training (no Vulkan SDK): `examples/nn_cifar10_f32_tiny_synth/`.

See [DEV_LIGHTWEIGHT.md](../../docs/DEV_LIGHTWEIGHT.md).
