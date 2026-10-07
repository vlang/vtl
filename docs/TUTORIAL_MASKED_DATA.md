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
column_sums := data.sum_along_axis(0, false)!
row_means := data.mean_along_axis(1, true)!
```

Global `sum` and `mean` return `MaskedValue`, whose `is_masked` field is true
when no valid values contributed. Axis reductions return another `MaskedArray`;
an output mask entry is true when its input slice had no valid values. The
underlying mean value for such a slice is NaN. Negative axis indices are
supported, and `keepdims` retains the reduced dimension with size one.

This API currently covers filling, compression, counting, and sum/mean only.
It does not yet implement masked elementwise operations or other mask-aware
statistics. NaN values are not implicitly treated as missing; use an explicit
boolean mask when that is desired.

See the [NumPy parity tracker](./NUMPY_PARITY.md) for remaining work.
