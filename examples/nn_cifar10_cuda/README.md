# nn_cifar10_cuda

Tiny synthetic CIFAR-shaped pipeline for **opt-in** GPU linear forward.

## Run (local, CUDA build)

```bash
export VTL_USE_CUDA=1
v -d cuda run vtl/examples/nn_cifar10_cuda/main.v
```

Without `VTL_USE_CUDA`, the example still runs on CPU (same weights as
`nn_cifar10_tiny_synth`).

## Notes

- Not executed in default CI (CUDA link + memory).
- For full CIFAR-10 + checkpoints see `examples/nn_cifar10/`.
- Device session staging: `autograd/device_session_test.v` with `VTL_TEST_CUDA=1`.
