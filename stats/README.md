# `vtl.stats`

Descriptive statistics and reductions over tensor elements. The API includes
sum/product, mean and means, median/mode, population and sample variance and
standard deviation, extrema and indices, covariance, quantiles, skewness,
kurtosis, and lag-one autocorrelation. Axis helpers are also available for
selected reductions.

`prod` follows the multiplicative identity convention: an empty input reduces
to `1`, while an empty sum reduces to `0`.

`sum_along_axis` and `product_along_axis` return tensors for a selected axis.
Pass `true` for `keepdims` to retain that axis with length one; negative axes
are accepted. Empty reduction axes return the corresponding identity for every
output slice.

`average_axis` keeps its reduced axis with length one. Use
`average_along_axis(..., keepdims)` to choose whether to keep or remove it;
weights may match the input or be a vector matching the selected axis.
`average_along_axes` reduces several axes with explicit `keepdims` and requires
weights with the same shape as the input.

`trapezoid`, `trapezoid_axis`, and `trapezoid_x_axis` integrate numeric tensors
with the composite trapezoidal rule and return `f64` tensors with the reduced
axis removed. Explicit sample coordinates remain in their given order.

`gradient_axis` estimates the numerical derivative along one axis using
uniform sample spacing. It preserves the input shape, accepts negative axes,
uses centered differences for interior points, and uses first-order one-sided
differences at the boundaries. `gradient_axis_with_coordinates` accepts
strictly monotonic non-uniform sample coordinates and uses first-order
boundary differences. `gradient_axis_with_coordinates_edge_order` selects
first- or second-order boundary differences explicitly. All gradient functions
preserve the input shape and support negative axes. See the
[gradient example](../examples/gradient).

NaN-aware reductions `nansum`, `nanprod`, `nanmin`, and `nanmax` ignore NaN
values and promote results to `f64`. Their `_axis` variants reduce one axis
and remove it; `_axis_keepdims` variants retain it with length one. Empty or
all-NaN sums and products use identities `0` and `1`. Empty or all-NaN minima
and maxima return NaN.

`histogram(data, bins)` infers a finite range from any numeric tensor and
returns a `Histogram` with `counts` and `bin_edges`. Empty input uses `[0, 1]`;
constant input expands by `0.5` at each end. `histogram_range(data, bins,
minimum, maximum)` uses explicit finite increasing bounds, ignores values
outside them, and includes the final right edge, matching NumPy's convention.
`histogram_auto(data, rule)` chooses a bin count using Sturges, Doane,
square-root, Rice, Scott, Freedman-Diaconis, Stone, or the NumPy-style
automatic maximum of Sturges and Freedman-Diaconis. Stone minimizes the
cross-validated integrated squared error over candidate counts up to
`max(100, floor(sqrt(n)))`. Degenerate estimates fall back to Sturges.
For constant input, the Stone rule returns one bin over the expanded range.

`bincount(input, minlength)` counts non-negative integer labels in a vector;
`bincount_weighted(input, weights, minlength)` sums numeric weights for each
label. Both return vectors sized to at least `minlength`.

```v
import vtl
import vtl.stats

values := vtl.from_1d[f64]([1.0, 2.0, 3.0, 4.0])!
average := stats.mean[f64](values)
spread := stats.sample_stddev[f64](values)
row_total := stats.sum_along_axis[f64](values, -1, true)!
```

The legacy `quantile` helper expects sorted input. The `*_with_method` APIs
sort a copy and support all thirteen NumPy estimators through
`QuantileMethod`, including global, multi-quantile, axis, percentile, and
NaN-ignoring forms. Axis quantiles accept `keepdims`; multi-quantile results
prepend the quantile dimension. See [`stats.v`](stats.v) and
[`quantile_methods.v`](quantile_methods.v) for edge-case behavior and the
[reductions tutorial](../docs/TUTORIAL_REDUCTIONS.md).

Weighted quantiles and percentiles use NumPy's `inverted_cdf` estimator.
Global weights match the input shape; one-axis forms accept full-shape weights
or a vector for the reduced axis. Multi-axis forms accept full-shape weights or
a compact tensor whose dimensions follow the supplied axes order. The
`nanquantile_weighted*` and `nanpercentile_weighted*` variants omit NaN values
and their corresponding weights.

`median` accepts unsorted tensors, uses linear interpolation, and returns
`f64` for every input type. An empty tensor returns NaN.

`nanmedian` ignores NaN values and returns NaN for empty or all-NaN inputs.
`nanmedian_axis` keeps the reduced dimension; `nanmedian_axis_squeeze` removes
it.

`mean_along_axes`, `variance_along_axes`, and `std_along_axes` reduce multiple
axes in one pass, with matching NaN-ignoring `nan*_along_axes` variants. All
accept `keepdims`; variance and standard deviation accept `ddof`.

`percentile_axis` and `nanpercentile_axis` accept percentiles on the 0..100
scale. Their `_squeeze` variants remove the reduced axis; the base variants
keep it with length one.
`percentiles_linear` and `percentiles_axis` accept multiple requested values
and prepend the percentile dimension to their results; `nanpercentiles_*`
counterparts ignore NaNs.
