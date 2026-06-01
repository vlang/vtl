# nn_cifar10_f32_tiny_synth

Synthetic mini-CIFAR batches in **f32**: `Sequential` forward, MSE, `backprop()`, and **Adam**.

No dataset download.
Complements `nn_cifar10_tiny_synth` (f64) and `nn_cifar10_vulkan` (forward-only).

## Run

```sh
v run vtl/examples/nn_cifar10_f32_tiny_synth/main.v
```

## Notes

- CPU only (Vulkan/CUDA examples stay in sibling folders).
- Scoped test: `nn/f32_training_smoke_test.v`.

See [DEV_LIGHTWEIGHT.md](../../docs/DEV_LIGHTWEIGHT.md).
