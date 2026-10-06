# Tensor axis manipulation

This example moves tensor axes without copying their underlying data, reverses
values along a selected axis with `Tensor.flip`, repeats values with
`vtl.repeat_axis`, and repeats full blocks with `vtl.tile`.

Run from `~/.vmodules`:

```sh
v run vtl/examples/tensor_axis_manipulation/main.v
```

See [the slicing tutorial](../../docs/TUTORIAL_SLICING.md#moving-axes) for the
API details and negative-axis rules. `flip` returns a tensor copy; pass no axes
to reverse every dimension, or pass one or more positive or negative axes.
`vtl.repeat` repeats values in flattened order, while `vtl.repeat_axis` preserves
the tensor rank and repeats each value along the requested axis.
`vtl.tile` repeats whole blocks and aligns repetition dimensions from the right.
`vtl.rot90` rotates the first two dimensions; use `vtl.rot90_k` for a chosen
number of quarter turns or `vtl.rot90_axes` to select another plane.
