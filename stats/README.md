# `vtl.stats`

Descriptive statistics and reductions over tensor elements. The API includes
sum/product, mean and means, median/mode, population and sample variance and
standard deviation, extrema and indices, covariance, quantiles, skewness,
kurtosis, and lag-one autocorrelation. Axis helpers are also available for
selected reductions.

`prod` follows the multiplicative identity convention: an empty input reduces
to `1`, while an empty sum reduces to `0`.

`trapezoid`, `trapezoid_axis`, and `trapezoid_x_axis` integrate numeric tensors
with the composite trapezoidal rule and return `f64` tensors with the reduced
axis removed. Explicit sample coordinates remain in their given order.

`histogram(data, bins)` infers a finite range from any numeric tensor and
returns a `Histogram` with `counts` and `bin_edges`. Empty input uses `[0, 1]`;
constant input expands by `0.5` at each end. `histogram_range(data, bins,
minimum, maximum)` uses explicit finite increasing bounds, ignores values
outside them, and includes the final right edge, matching NumPy's convention.

`bincount(input, minlength)` counts non-negative integer labels in a vector;
`bincount_weighted(input, weights, minlength)` sums numeric weights for each
label. Both return vectors sized to at least `minlength`.

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
