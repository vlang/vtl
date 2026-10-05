# `vtl.stats`

Descriptive statistics and reductions over tensor elements. The API includes
sum/product, mean and means, median/mode, population and sample variance and
standard deviation, extrema and indices, covariance, quantiles, skewness,
kurtosis, and lag-one autocorrelation. Axis helpers are also available for
selected reductions.

```v
import vtl
import vtl.stats

values := vtl.from_1d[f64]([1.0, 2.0, 3.0, 4.0])!
average := stats.mean[f64](values)
spread := stats.sample_stddev[f64](values)
```

`quantile` currently expects its tensor argument to be sorted. See function
comments in [`stats.v`](stats.v) for edge-case behavior and the
[reductions tutorial](../docs/TUTORIAL_REDUCTIONS.md).
