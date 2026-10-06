# Reductions — argmax, argmin, cumsum, cumprod

VTL provides several reduction operations that summarise a tensor along one or more axes.

## argmax / argmin

`argmax_axis(axis)` returns the index of the maximum value along `axis`.
`argmin_axis(axis)` returns the index of the minimum value.

If no axis is specified, the tensor is flattened first.

```v
import vtl

// 2-D tensor: rows = [3,5], [1,4]
t := vtl.from_array[f64]([3.0, 5.0, 1.0, 4.0], [2, 2])!

// Along axis 1 (columns): which column holds the max per row?
amax := t.argmax_axis[f64](1)!
assert amax.shape == [2]
// amax = [1, 1]  → row 0: max is at col 1 (5.0), row 1: max is at col 1 (4.0)

amin := t.argmin_axis[f64](0)!
assert amin.shape == [2]
// amin = [1, 0]  → col 0: min is at row 1 (1.0), col 1: min is at row 0 (4.0)

// Global: index of the largest element (no axis)
global_max := t.argmax[f64](0)!
// global_max = 1  → t.data[1] == 5.0 is the largest element
```

## max / min per axis

`max_axis(axis)` returns a tensor containing the maximum value per slice along `axis`.
`min_axis` does the same for the minimum.

```v
import vtl

t := vtl.from_array[f64]([3.0, 5.0, 1.0, 4.0], [2, 2])!

mx := t.max_axis[f64](1)!
// mx = [5.0, 4.0]

mn := t.min_axis[f64](0)!
// mn = [1.0, 4.0]
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

`vtl.stats.percentile_linear` accepts the equivalent 0..100 percentile scale.
`vtl.stats.quantile_axis` computes one quantile per axis slice and retains the
reduced axis with length one.

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

values := vtl.from_1d([30.0, 0.0, 20.0, 10.0])!
median := stats.quantile_linear(values, 0.5)! // 15.0
median_percentile := stats.percentile_linear(values, 50)! // 15.0
rows := vtl.from_array([1.0, 3.0, 5.0, 7.0], [2, 2])!
row_medians := stats.quantile_axis(rows, 0.5, 1)! // shape [2, 1]
```

Empty tensors and out-of-range quantiles return errors. NaNs propagate to the
result.

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
