# nn_cifar10_tiny_synth

Synthetic CIFAR-shaped batches (no dataset download). **Default CI smoke.**

## Run

```sh
v run vtl/examples/nn_cifar10_tiny_synth/main.v
```

With CUDA Linear forward (local GPU, `-d cuda` build):

```sh
export VTL_USE_CUDA=1
v -d cuda run vtl/examples/nn_cifar10_tiny_synth/main.v
```

Phase 2 GPU activation chain (skips input H2D between Linear layers):

```sh
export VTL_USE_CUDA=1
export VTL_GPU_ACTIVATIONS=1
v -d cuda run vtl/examples/nn_cifar10_tiny_synth/main.v
```

See [DEV_LIGHTWEIGHT.md](../../docs/DEV_LIGHTWEIGHT.md).
