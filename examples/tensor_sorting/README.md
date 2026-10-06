# Tensor sorting

This example sorts tensors along an axis and returns stable sort indices. NaN
values sort after all numeric values.

Run from `~/.vmodules`:

```sh
v run vtl/examples/tensor_sorting/main.v
```

`vtl.sort` and `vtl.argsort` operate along the last axis. Use `vtl.sort_axis`
and `vtl.argsort_axis` to select another axis. Sort indices are local to the
selected axis.
