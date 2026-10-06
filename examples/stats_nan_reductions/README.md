# NaN-aware reductions

Run from `~/.vmodules`:

```sh
v run ./vtl/examples/stats_nan_reductions/main.v
```

This example skips missing `NaN` samples for global sum/product and per-row
minimum/maximum reductions. Axis helpers return `f64` tensors and provide
`_axis_keepdims` variants when the reduced dimension should remain in the
result shape.
