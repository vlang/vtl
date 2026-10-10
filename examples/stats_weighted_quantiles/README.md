# Weighted quantiles

This example computes a flattened weighted median, row-wise weighted medians,
and row-wise weighted quantiles while retaining the reduced axis. VTL uses the
inverted-CDF estimator for weighted quantiles, matching NumPy's supported
weighted method.

Run from `~/.vmodules`:

```bash
v run ./vtl/examples/stats_weighted_quantiles/main.v
```
