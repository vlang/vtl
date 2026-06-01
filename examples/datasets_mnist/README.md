# datasets_mnist

Shape smoke for the MNIST dataset loader.

## Run

```bash
v run vtl/examples/datasets_mnist/main.v
```

The loader returns:

- `train_features`: `[60000, 28, 28]`
- `test_features`: `[10000, 28, 28]`
- `train_labels`: `[60000]`
- `test_labels`: `[10000]`

This example may download/cache dataset files depending on the local dataset
state. For CI-safe training, prefer synthetic CIFAR examples.

See [datasets/README.md](../../datasets/README.md).
