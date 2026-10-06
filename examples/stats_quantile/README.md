# Quantiles and NaN-aware statistics

This example calculates linearly interpolated quantiles and percentiles,
per-axis quantiles, and NaN-aware mean and standard deviation.

Run from `~/.vmodules`:

```sh
v run vtl/examples/stats_quantile/main.v
```

The NaN-aware axis functions retain the reduced axis with length one. Slices
with no non-NaN values return NaN.
