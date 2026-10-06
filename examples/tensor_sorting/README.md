# Tensor sorting

This example sorts tensors along an axis, returns stable sort indices, and
shows partial value/index ordering with `partition` and `argpartition`. NaN
values sort after all numeric values.

Run from `~/.vmodules`:

```sh
v run vtl/examples/tensor_sorting/main.v
```

`vtl.sort` and `vtl.argsort` operate along the last axis. Use `vtl.sort_axis`
and `vtl.argsort_axis` to select another axis. Sort indices are local to the
selected axis. `vtl.partition` and `vtl.argpartition` place a requested kth
value at its sorted position without fully ordering each slice; use their
`_axis` forms and a list of kth positions for other axes or multiple cut points.
