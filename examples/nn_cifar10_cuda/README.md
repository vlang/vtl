# nn_cifar10_cuda

Synthetic mini-CIFAR f64 training: Flatten + Linear + MSE,
`backprop()`, Adam, and `DeviceSession` via `autograd_cuda.attach_context_session`.

## Run (local, CUDA build)

```bash
export VTL_USE_CUDA=1
v -d cuda run vtl/examples/nn_cifar10_cuda/main.v
```

Phase 2 GPU activation chain between Linear layers:

```bash
export VTL_USE_CUDA=1
export VTL_GPU_ACTIVATIONS=1
v -d cuda run vtl/examples/nn_cifar10_cuda/main.v
```

Without `VTL_USE_CUDA`, the example still runs on CPU (same shape as this smoke).

## Test

```bash
VTL_USE_CUDA=1 VTL_TEST_CUDA=1 VJOBS=1 v -d cuda test vtl/nn/cuda_training_smoke_test.v
```

## Notes

- Not executed in default CI (CUDA link + memory).
- Spatial size 8×8 (light smoke; 16×16+attach can stack-overflow on CPU backprop).
- f32 training: `examples/nn_cifar10_f32_tiny_synth/`.

See [DEVICE_MEMORY.md](../../docs/DEVICE_MEMORY.md).

See [DEV_LIGHTWEIGHT.md](../../docs/DEV_LIGHTWEIGHT.md).
