# Tensor axis manipulation

This example moves tensor axes without copying their underlying data and reverses
the order of values along a selected axis with `Tensor.flip`.

Run from `~/.vmodules`:

```sh
v run vtl/examples/tensor_axis_manipulation/main.v
```

See [the slicing tutorial](../../docs/TUTORIAL_SLICING.md#moving-axes) for the
API details and negative-axis rules. `flip` returns a tensor copy; pass no axes
to reverse every dimension, or pass one or more positive or negative axes.
