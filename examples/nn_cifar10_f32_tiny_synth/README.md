# nn_cifar10_f32_tiny_synth

Synthetic mini-CIFAR batches in **f32**: `Sequential` forward, MSE, `backprop()`, and **Adam**.

No dataset download.
Vulkan Linear forward: `examples/nn_cifar10_f32_vulkan_tiny_synth/`.

## Run

```sh
v run vtl/examples/nn_cifar10_f32_tiny_synth/main.v
```

## Notes

- CPU only (Vulkan/CUDA examples stay in sibling folders).
- Scoped test: `nn/f32_training_smoke_test.v`.

See [DEV_LIGHTWEIGHT.md](../../docs/DEV_LIGHTWEIGHT.md).
