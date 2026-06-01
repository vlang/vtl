# nn_cifar10_vulkan

Tiny synthetic CIFAR-shaped **f32** forward smoke. **Linear** uses Vulkan GEMM when opted in.

## Run

```bash
# CPU Linear (no Vulkan SDK required)
v run vtl/examples/nn_cifar10_vulkan/main.v

# GPU Linear via Vulkan
VTL_USE_VULKAN=1 v -d vulkan run vtl/examples/nn_cifar10_vulkan/main.v
```

Full f32 training (loss, backprop, Adam) is not in this example yet.
Only forward Linear is GPU-accelerated when opted in.

## Notes

- Not run in default CI (Vulkan SDK + GPU).
- Requires `vsl` with `-d vulkan`.
- For CUDA opt-in see `examples/nn_cifar10_cuda/`.
