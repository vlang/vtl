# nn_cifar10_vulkan

CIFAR-shaped **f32** training smoke (`[3,8,8]` → Conv2D → flatten → Linear → MSE). **Linear** and **Conv2D** (no padding) use Vulkan when opted in; backward stays CPU for Conv2D.

## Run

```bash
# CPU Linear (no Vulkan SDK required)
v run vtl/examples/nn_cifar10_vulkan/main.v

# GPU Linear via Vulkan (use -prod on V 0.5.1+ to avoid debug-instance crash)
VTL_USE_VULKAN=1 v -prod -d vulkan run vtl/examples/nn_cifar10_vulkan/main.v
```

Smaller synthetic variant: `examples/nn_cifar10_f32_vulkan_tiny_synth/`.

## Notes

- Not run in default CI (Vulkan SDK + GPU).
- Requires `vsl` with `-d vulkan`.
- For CUDA opt-in see `examples/nn_cifar10_cuda/`.
