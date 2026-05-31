# nn_cifar10_vulkan

Tiny synthetic CIFAR-shaped **f32** pipeline with an explicit `linear_forward_vulkan` smoke step.

## Run

```bash
v -d vulkan run vtl/examples/nn_cifar10_vulkan/main.v
```

Training uses CPU autograd; Vulkan is validated via a small GEMM linear forward at the end.

## Notes

- Not run in default CI (Vulkan SDK + GPU).
- Requires `vsl` with `-d vulkan`.
- For CUDA opt-in see `examples/nn_cifar10_cuda/`.
