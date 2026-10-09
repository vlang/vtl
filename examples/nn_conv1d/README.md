# Conv1D forward and autograd

Build a small sequential model with a padded Conv1D layer and `tanh`, run a sequence through it, then backpropagate to inspect the input gradient shape.

## Run

From `~/.vmodules`:

```sh
v run ./vtl/examples/nn_conv1d/main.v
```

## Notes

The input has shape `[batch, channels, length]`. Weights are initialized by the model, so printed values may vary.
