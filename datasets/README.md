# VTL Datasets

VTL provides dataset loaders and batching utilities for ML examples and tests.

## Available datasets

| Dataset | Loader | Purpose |
|---------|--------|---------|
| MNIST | `datasets.load_mnist()` | Handwritten digit images (`28x28`) |
| IMDB | `datasets.load_imdb()` | Sentiment analysis reviews |
| CIFAR-10 | `datasets.load_cifar10(...)` | Image classification examples |
| DataLoader | `datasets.DataLoader[T]` | Batch, shuffle, and iterate tensors/labels |

## Examples

Run from `~/.vmodules`:

```bash
v run vtl/examples/datasets_mnist/main.v
v run vtl/examples/datasets_imdb/main.v
v run vtl/examples/nn_cifar10_tiny_synth/main.v
```

Use synthetic examples (`nn_cifar10_tiny_synth`,
`nn_cifar10_f32_tiny_synth`) for CI and quick local checks. Real dataset
examples may download/cache data and should be treated as local integration
tests.

## Related docs

- [Examples catalog](../examples/README.md)
- [Neural network tutorial](../docs/TUTORIAL_NEURAL_NETWORKS.md)
- [Lightweight development commands](../docs/DEV_LIGHTWEIGHT.md)
