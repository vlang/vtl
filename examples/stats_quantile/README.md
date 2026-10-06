# Quantiles and NaN-aware statistics

This example calculates linearly interpolated quantiles and demonstrates
NaN-aware mean and standard deviation, including a per-axis mean.

Run from `~/.vmodules`:

```sh
v run vtl/examples/stats_quantile/main.v
```

The NaN-aware axis functions retain the reduced axis with length one. Slices
with no non-NaN values return NaN.
