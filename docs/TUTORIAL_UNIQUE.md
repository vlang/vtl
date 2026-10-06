# Tutorial: Unique Values

`vtl.unique` flattens a tensor and returns its distinct values in ascending
order. `vtl.unique_counts` also returns the number of input occurrences for
each sorted value.

```v
import vtl

samples := vtl.from_2d([[4, 2, 4], [1, 2, 4]])!
values := vtl.unique(samples)!
summary := vtl.unique_counts(samples)!

assert values.to_array() == [1, 2, 4]
assert summary.values.to_array() == [1, 2, 4]
assert summary.counts.to_array() == [1, 2, 3]
inverse := vtl.unique_inverse(samples)!
assert inverse.to_array() == [2, 1, 2, 0, 1, 2]
first_indices := vtl.unique_first_indices(samples)!
assert first_indices.to_array() == [3, 1, 0]
```

`unique_inverse` returns one index per flattened input value. Use those indices
to map each source value back into the sorted unique-value array.
`unique_first_indices` returns the first flattened input position for each
sorted unique value.

For row, column, or higher-rank slices, use `unique_axis` or
`unique_axis_result`. The result API returns values with the selected axis
replaced by the number of unique slices, plus counts, first source positions,
and inverse indices for that axis:

```v
rows := vtl.from_2d([[2, 1], [1, 4], [2, 1]])!
unique_rows := vtl.unique_axis_result(rows, 0)!
assert unique_rows.values.to_array() == [1, 4, 2, 1]
assert unique_rows.counts.to_array() == [1, 2]
assert unique_rows.first_indices.to_array() == [1, 0]
assert unique_rows.inverse.to_array() == [1, 0, 1]
```

The returned values are sorted regardless of input order. For `f32` and `f64`,
all NaN values are grouped into one unique value and placed after finite values,
matching VTL's sort ordering. Empty tensors return empty values, counts, inverse
indices, and first indices.

Run the complete example from `~/.vmodules`:

```bash
v run ./vtl/examples/unique/main.v
```
