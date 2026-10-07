# Reductions — argmax, argmin, cumsum, cumprod

VTL provides several reduction operations that summarise a tensor along one or more axes.

## Discrete differences

`vtl.diff` computes adjacent differences along an axis. Increase `n` to apply
the operation repeatedly, and use a negative axis to count from the end:

```v
import vtl

series := vtl.from_1d([1.0, 2.0, 4.0, 7.0])!
first_difference := vtl.diff[f64](series, 1, -1)!
second_difference := vtl.diff[f64](series, 2, -1)!
assert first_difference.to_array() == [1.0, 2.0, 3.0]
assert second_difference.to_array() == [1.0, 1.0]
```

## Trapezoidal integration

`vtl.stats.trapezoid` integrates over the last axis with uniform spacing.
Use `trapezoid_axis` for another axis, or `trapezoid_x_axis` for explicit
coordinates. The order of explicit coordinates is preserved, so decreasing
coordinates produce a negative integral, matching
[NumPy's `trapezoid`](https://numpy.org/doc/stable/reference/generated/numpy.trapezoid.html).

```v
import vtl
import vtl.stats

series := vtl.from_1d([1.0, 2.0, 3.0])!
assert stats.trapezoid[f64](series, 1.0)!.get_nth(0) == 4.0
```

## Reduce several axes

Use `sum_along_axes` or `product_along_axes` when one reduction should collapse
multiple dimensions. Axes may be negative, and `keepdims` preserves each
reduced dimension with length one:

```v
import vtl
import vtl.stats

values := vtl.from_array([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2])!
sums := stats.sum_along_axes[int](values, [0, -1], false)!
assert sums.shape == [2]
assert sums.to_array() == [14, 22]

products := stats.product_along_axes[int](values, [0, -1], true)!
assert products.shape == [1, 2, 1]
assert products.to_array() == [60, 672]
```

## Statistical reductions along an axis

`mean_along_axis`, `variance_along_axis`, and `std_along_axis` accept an
explicit `keepdims` argument. Their `nanmean_`, `nanvar_`, and `nanstd_`
counterparts ignore NaN values. Variance and standard deviation also accept
`ddof`, the delta subtracted from the sample count:

```v
import vtl
import vtl.stats

values := vtl.from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])!
means := stats.mean_along_axis(values, 1, false)!
assert means.shape == [2]
assert means.to_array() == [2.0, 5.0]

sample_std := stats.std_along_axis(values, 0, 1, true)!
assert sample_std.shape == [1, 3]
assert sample_std.to_array() == [2.1213203435596424, 2.1213203435596424, 2.1213203435596424]
```

NaN-aware sum, product, minimum, and maximum also support multiple axes and
`keepdims` through `nansum_axes`, `nanprod_axes`, `nanmin_axes`, and
`nanmax_axes`. Slices with no non-NaN values yield NaN for min/max.

```v
import math
import vtl
import vtl.stats

volume := vtl.from_array[f64]([math.nan(), 2.0, 3.0, 4.0, 5.0, math.nan(), 7.0, 8.0], [
	2,
	2,
	2,
])!
mins := stats.nanmin_axes(volume, [0, -1], false)!
assert mins.to_array() == [2.0, 3.0]
```

## argmax / argmin

`argmax_axis(axis)` and `argmin_axis(axis)` retain the reduced axis with
length one. Use `argmax_axis_squeeze(axis)` or `argmin_axis_squeeze(axis)` to
remove it, matching NumPy's default `keepdims=false` shape behavior. VTL's
one-dimensional tensors use shape `[1]` for scalar results.

If no axis is specified, the tensor is flattened first.

```v
import vtl

// 2-D tensor: rows = [3,5], [1,4]
t := vtl.from_array[f64]([3.0, 5.0, 1.0, 4.0], [2, 2])!

// Along axis 1 (columns): which column holds the max per row?
amax := t.argmax_axis_squeeze[f64](1)!
assert amax.shape == [2]
// amax = [1, 1]  → row 0: max is at col 1 (5.0), row 1: max is at col 1 (4.0)

amin := t.argmin_axis_squeeze[f64](0)!
assert amin.shape == [2]
// amin = [1, 0]  → col 0: min is at row 1 (1.0), col 1: min is at row 0 (4.0)

// Global: index of the largest element (no axis)
global_max := t.argmax[f64](0)!
// global_max = 1  → t.data[1] == 5.0 is the largest element
```

## max / min per axis

`max_axis(axis)` and `min_axis(axis)` retain the reduced axis with length one.
Use `max_axis_squeeze(axis)` or `min_axis_squeeze(axis)` to remove it, matching
NumPy's default `keepdims=false` shape behavior. VTL represents scalar-shaped
reductions as shape `[1]`. Use `max_axes(axes, keepdims)` or
`min_axes(axes, keepdims)` to reduce multiple unique axes in one call.

```v
import vtl

t := vtl.from_array[f64]([3.0, 5.0, 1.0, 4.0], [2, 2])!

mx := t.max_axis_squeeze[f64](1)!
// mx = [5.0, 4.0]

mn := t.min_axis_squeeze[f64](0)!
// mn = [1.0, 4.0]

volume := vtl.from_array[f64]([0.0, 5.0, 3.0, 7.0, 4.0, 2.0, 8.0, 1.0], [2, 2, 2])!
assert volume.max_axes([0, -1], false)!.to_array() == [5.0, 8.0]
```

## Logical all / any along an axis

`all_axis` and `any_axis` treat numeric zero as false and nonzero values as
true. Pass `true` as the second argument to retain the reduced axis.
Empty reductions return the logical identities: `all_axis` returns true and
`any_axis` returns false. Use `all_axes` and `any_axes` to reduce several
unique axes at once; negative axes are supported.
An empty axis list converts values to booleans without reducing dimensions,
matching NumPy's `axis=()` behavior.

```v
import vtl

values := vtl.from_array([0, 1, 2, 0, 3, 4], [2, 3])!
assert values.all_axis(1, false)!.to_array() == [false, false]
assert values.any_axis(1, true)!.shape == [2, 1]
assert values.any_axis(1, false)!.to_array() == [true, true]

volume := vtl.from_array([0, 1, 2, 3, 0, 4, 5, 6], [2, 2, 2])!
assert volume.all_axes([0, -1], false)!.to_array() == [false, true]
```

## cumsum / cumprod

`cumsum(axis)` computes the cumulative sum along `axis`.
`cumprod(axis)` computes the cumulative product.

```v
import vtl

t := vtl.from_array[f64]([1.0, 2.0, 3.0, 4.0], [4])!

cs := t.cumsum[f64](0)!
// cs = [1.0, 3.0, 6.0, 10.0]

cp := t.cumprod[f64](0)!
// cp = [1.0, 2.0, 6.0, 24.0]
```

For 2-D tensors the same API works along either axis:

```v
import vtl

t2 := vtl.from_array[f64]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])!
// t2 = [[1,2,3],
//       [4,5,6]]

t2.cumsum[f64](1)!
// = [[1, 3, 6],
//    [4, 9, 15]]   — cumulative sum along rows (axis=1)

t2.cumsum[f64](0)!
// = [[1, 2,  3],
//    [5, 7,  9]]   — cumulative sum along columns (axis=0)
```

## Autograd support

All reduction operations above are differentiable when called through a
`Variable`. See [TUTORIAL_AUTOGRAD.md](./TUTORIAL_AUTOGRAD.md) for general
information about automatic differentiation in VTL.

## Quantiles

`vtl.stats.quantile_linear` sorts a copy of the tensor values and computes
NumPy's default linearly interpolated quantile. The quantile must be between
0 and 1, and the result is `f64` even for integer input tensors.

Use `quantiles_linear` to calculate several quantiles from one sorted copy:

```v
import vtl
import vtl.stats

values := vtl.from_1d([30.0, 0.0, 20.0, 10.0])!
quartiles := stats.quantiles_linear(values, [0.0, 0.25, 0.5, 0.75, 1.0])!
assert quartiles.to_array() == [0.0, 7.5, 15.0, 22.5, 30.0]
```

Use `quantiles_axis` to compute several quantiles for each axis slice. Its
result puts the quantile dimension first and removes the reduced axis, matching
NumPy's default `quantile` shape. Each slice is sorted once for all requested
quantiles:

```v
import vtl
import vtl.stats

grid := vtl.from_array([9.0, 1.0, 8.0, 2.0, 7.0, 3.0], [2, 3])!
quartiles_by_row := stats.quantiles_axis(grid, [0.25, 0.5, 0.75], 1)!
assert quartiles_by_row.shape == [3, 2]
```

`vtl.stats.percentile_linear` accepts the equivalent 0..100 percentile scale.
`vtl.stats.quantile_axis` computes one quantile per axis slice and retains the
reduced axis with length one. Ordinary quantiles propagate NaNs.
Use `quantile_axis_squeeze` or `nanquantile_axis_squeeze` to remove the axis
instead, matching NumPy's output shape for a single quantile.
The `nanquantile_linear`, `nanquantiles_linear`, `nanpercentile_linear`, and
`nanquantile_axis` variants ignore NaN values; `nanquantiles_axis` returns
multiple NaN-aware values per slice. A slice containing only NaNs returns NaN
for each requested quantile. Ordinary axis quantiles propagate NaNs within
their slice.

## Stable variance and standard deviation

`stats.variance` uses Welford's online algorithm and returns `f64`, including
for integer tensors. Its `ddof` parameter selects population variance (`0`, the
default) or sample variance (`1`). The function returns an error when the
effective denominator is not positive.

`mean_axis`, `variance_axis`, and `std_axis` apply the same reductions per
axis slice and retain the reduced dimension with length one. `variance_axis`
and `std_axis` accept the same `ddof` convention. NaNs propagate in the
ordinary reductions; use the `nan*` variants to ignore them.

```v
import vtl
import vtl.stats
import math

values := vtl.from_1d([30.0, 0.0, 20.0, 10.0])!
median := stats.median(values) // 15.0; input does not need sorting
median_percentile := stats.percentile_linear(values, 50)! // 15.0
quartiles := stats.percentiles_linear(values, [25, 50, 75])!
integer_median := stats.median(vtl.from_1d([1, 2])!) // 1.5
rows := vtl.from_array([1.0, 3.0, 5.0, 7.0], [2, 2])!
row_medians := stats.quantile_axis(rows, 0.5, 1)! // shape [2, 1]
row_percentiles := stats.percentile_axis(rows, 50, 1)! // shape [2, 1]
row_quartiles := stats.percentiles_axis(rows, [25, 50, 75], 1)! // shape [3, 2]

measurements := vtl.from_array([1.0, math.nan(), 3.0, 5.0, math.nan(), math.nan()], [
	3,
	2,
])!
column_medians := stats.nanquantile_axis(measurements, 0.5, 0)! // [[2.0, 5.0]]
overall_nanmedian := stats.nanmedian(measurements) // 3.0
column_nanmedians := stats.nanmedian_axis_squeeze(measurements, 0)! // [2.0, 5.0]
column_nanpercentiles := stats.nanpercentile_axis_squeeze(measurements, 50, 0)! // [2.0, 5.0]
```

Empty tensors and out-of-range quantiles return errors. The NaN-aware variants
return NaN when all values in a reduction slice are NaN.
`stats.median` returns NaN for an empty tensor and returns `f64`, including
for integer inputs with a fractional median.

```v
import vtl
import vtl.stats

samples := vtl.from_1d([1, 2, 3])!
population := stats.variance(samples, stats.VarianceData{})!
sample := stats.variance(samples, stats.VarianceData{
	ddof: 1
})!
deviation := stats.std(samples, stats.VarianceData{})!
assert population > 0.66 && population < 0.67
assert sample == 1.0
assert deviation > 0.81 && deviation < 0.82
```

## Weighted averages

`stats.average` calculates one weighted mean when values and weights have the
same shape. `stats.average_axis` accepts either full-shaped weights or a
one-dimensional weight vector matching the selected axis; the result keeps
that axis with length one.

```v
import vtl
import vtl.stats

measurements := vtl.from_2d([[4.0, 5.0, 4.5], [3.0, 4.0, 5.0]])!
reliability := vtl.from_1d([1.0, 2.0, 1.0])!
row_means := stats.average_axis(measurements, reliability, 1)!
assert row_means.to_array() == [4.625, 4.0]
```

Empty input, incompatible shapes, or a zero sum of weights returns an error.

## NaN-aware sums and extrema

`nansum`, `nanprod`, `nanmin`, and `nanmax` skip NaN values. Their axis
variants return promoted `f64` tensors. Use `_axis_keepdims` to retain the
reduced axis with length one. Empty or all-NaN slices reduce to `0` for sums,
`1` for products, and NaN for minima or maxima.

```v
import vtl
import vtl.stats
import math

samples := vtl.from_array([1.0, math.nan(), 3.0, 4.0, math.nan(), 6.0], [2, 3])!
assert stats.nansum(samples) == 14.0
assert stats.nanprod(samples) == 72.0
assert stats.nanmin_axis(samples, 1)!.to_array() == [1.0, 4.0]
assert stats.nanmax_axis_keepdims(samples, 1)!.to_array() == [3.0, 6.0]
```
