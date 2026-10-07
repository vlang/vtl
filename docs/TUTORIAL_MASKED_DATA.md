# Masked data

`MaskedArray[T]` stores a tensor and a boolean mask with the same broadcasted
shape. A `true` mask entry means that value is missing, following NumPy's
`numpy.ma` convention. The values tensor is kept separately from the mask, so
masked payloads can be arbitrary and are ignored by supported reductions.

Run the complete example from `~/.vmodules`:

```bash
VJOBS=2 v run ./vtl/examples/masked_array/main.v
```

## Construct and inspect

```v
import vtl

measurements := vtl.from_array[f64]([12.0, 15.0, 18.0, 21.0], [2, 2])!
missing := vtl.from_array[bool]([false, true], [2])!
data := vtl.masked_array(measurements, missing)!

assert data.values.shape == [2, 2]
assert data.count() == 2
assert data.filled(-1.0).to_array() == [12.0, -1.0, 18.0, -1.0]
assert data.compressed()!.to_array() == [12.0, 18.0]
```

The row mask broadcasts across the leading dimension. `filled` returns a copy;
`compressed` returns a one-dimensional copy in row-major logical order.

## Reduce valid values

```v
total := data.sum()
average := data.mean()
product := data.prod()
minimum := data.min()
maximum := data.max()
sample_variance := data.variance(1)!
sample_deviation := data.std(1)!
column_sums := data.sum_along_axis(0, false)!
row_means := data.mean_along_axis(1, true)!
all_values_sum := data.sum_along_axes([0, 1], false)!
all_values_product := data.prod_along_axes([0, 1], false)!
all_values_variance := data.variance_along_axes([0, 1], 1, false)!
```

Global `sum` and `mean` return `MaskedValue`, whose `is_masked` field is true
when no valid values contributed. Axis reductions return another `MaskedArray`;
an output mask entry is true when its input slice had no valid values. The
underlying mean value for such a slice is NaN. Negative axis indices are
supported, and `keepdims` retains the reduced dimension with size one.
Global and axis-wise products use one as their identity; their output mask
still marks slices with no valid values. `min` and `max` return the extreme
valid values globally or across one or more axes, and mask any slice with no
valid values. The payload of a masked min/max result is unspecified and should
be ignored.
`variance` and `std` use Welford's online algorithm; `ddof=0` computes
population statistics and `ddof=1` computes sample statistics. Global or
per-slice results with `count <= ddof` are masked and carry a NaN payload.
The `*_along_axes` variants reduce a list of axes in one operation; negative
axes are supported and duplicates are rejected. An empty axes list preserves
values and masks elementwise (the mean variant converts values to `f64`).

This API currently covers filling, compression, counting, and sum/product/
minimum/maximum/mean/variance/standard deviation. It does not yet implement
masked elementwise operations or other mask-aware statistics. NaN values are
not implicitly treated as missing; use an explicit boolean mask when that is
desired.

See the [NumPy parity tracker](./NUMPY_PARITY.md) for remaining work.
