# nn_cifar10_f32_vulkan_tiny_synth

f32 **training** smoke (forward, MSE, backprop, Adam). **Linear** forward can use Vulkan GEMM.

## Run

```bash
# CPU Linear (default)
v run vtl/examples/nn_cifar10_f32_vulkan_tiny_synth/main.v

# Vulkan Linear forward
VTL_USE_VULKAN=1 v -d vulkan run vtl/examples/nn_cifar10_f32_vulkan_tiny_synth/main.v
```

## Test

```bash
VTL_TEST_VULKAN=1 VJOBS=1 v -d vulkan test vtl/nn/f32_vulkan_training_smoke_test.v
```

CPU-only f32 training (no Vulkan SDK): `examples/nn_cifar10_f32_tiny_synth/`.

See [DEV_LIGHTWEIGHT.md](../../docs/DEV_LIGHTWEIGHT.md).
