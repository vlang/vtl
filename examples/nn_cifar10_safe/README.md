# nn_cifar10_safe

CIFAR-10 training with conservative batch/epoch defaults for local dev. This
example uses a compact dense classifier so it compiles with modest memory; see
`nn_cifar10/` for the convolutional model.

## Run

```sh
v run ./vtl/examples/nn_cifar10_safe/main.v
```

For full training see `nn_cifar10/` (high compile RAM on some hosts).
