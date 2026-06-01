# nn_cifar10_vulkan

Tiny synthetic CIFAR-shaped **f32** forward smoke. **Linear** uses Vulkan GEMM when opted in.

## Run

```bash
# CPU Linear (no Vulkan SDK required)
v run vtl/examples/nn_cifar10_vulkan/main.v

# GPU Linear via Vulkan
VTL_USE_VULKAN=1 v -d vulkan run vtl/examples/nn_cifar10_vulkan/main.v
```

f32 training with optional Vulkan Linear forward:
`examples/nn_cifar10_f32_vulkan_tiny_synth/`.
This example only benchmarks forward Linear GEMM.

## Notes

- Not run in default CI (Vulkan SDK + GPU).
- Requires `vsl` with `-d vulkan`.
- For CUDA opt-in see `examples/nn_cifar10_cuda/`.
