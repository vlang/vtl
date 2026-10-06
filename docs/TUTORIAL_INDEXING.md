# Indexing, gather, and scatter

`take` and `take_along_axis` gather copies from a tensor. Negative axes and
indices count from the end. See [slicing](./TUTORIAL_SLICING.md) when a view of
a range is more appropriate.

## Search and digitize

`searchsorted` returns insertion positions in an ascending one-dimensional
tensor. Choose `.left` to insert before equal values or `.right` to insert
after them. The result has the shape of the query tensor.
`digitize` assigns values to monotonic bin edges, including decreasing edges;
`right` controls which interval owns an edge value.

```v
import vtl

sorted := vtl.from_1d([1, 3, 3, 5])!
queries := vtl.from_1d([0, 3, 4])!
assert vtl.searchsorted(sorted, queries, .left)!.to_array() == [0, 1, 3]
assert vtl.searchsorted(sorted, queries, .right)!.to_array() == [0, 3, 3]

measurements := vtl.from_1d([0.2, 1.5, 2.0, 3.8, 5.1])!
bins := vtl.from_1d([0.0, 2.0, 4.0, 6.0])!
assert vtl.digitize(measurements, bins, false)!.to_array() == [1, 1, 2, 2, 3]
```

`searchsorted` rejects values that are not sorted in ascending order. `digitize`
rejects non-monotonic bins.

## Find non-zero coordinates

`argwhere` groups each non-zero element's coordinates into one row. Its output
shape is `[number of matches, input rank]`; a scalar input therefore produces
one row with zero columns when it is non-zero.

```v
import vtl

values := vtl.from_2d([[0, 2, 0], [3, 4, 0]])!
coordinates := vtl.argwhere[int](values)!
assert coordinates.shape == [3, 2]
assert coordinates.to_array() == [0, 1, 1, 0, 1, 1]
```

As with NumPy's `argwhere`, these coordinate rows are useful for inspection and
coordinate-based updates; they are not a tuple of per-axis index arrays.
Use `nonzero` when you need one index tensor per axis:

```v
indices := vtl.nonzero[int](values)!
assert indices[0].to_array() == [0, 1, 1]
assert indices[1].to_array() == [1, 0, 1]
```

For a scalar input, `nonzero` returns one index tensor, using index `0` for a
nonzero value and an empty tensor for zero.

`put_along_axis` writes values into a mutable tensor. `scatter_add` adds values
to the selected locations; duplicate indices accumulate in row-major order.
Both operations require the index tensor and values tensor to have matching
shapes and ranks. Dimensions outside the selected axis may be smaller than the
target, and negative indices are supported.

```v
import vtl

mut values := vtl.from_array([1, 2, 3, 4, 5, 6], [2, 3])!
indices := vtl.from_array([1, 1, 0, 0], [2, 2])!
updates := vtl.from_array([10, 20, 30, 40], [2, 2])!

values.scatter_add(indices, updates, 1)!
assert values.to_array() == [1, 32, 3, 74, 5, 6]

values.put_along_axis(indices, updates, 1)!
assert values.to_array() == [1, 20, 3, 40, 5, 6]
```

`put_along_axis` applies repeated writes in row-major order, so the last value
for a repeated index wins. Both update operations validate all indices before
mutating the target. Indexed writes currently operate on tensors directly and
are not autograd operations.

Boolean masks passed to `masked_select` and `masked_fill` may broadcast to the
tensor shape. `masked_select` returns selected elements as a one-dimensional
copy; `masked_fill` returns a new tensor and leaves the input unchanged.

```v
import vtl

values := vtl.from_2d([[1, 2, 3], [4, 5, 6]])!
column_mask := vtl.from_1d([false, true, true])!
selected := values.masked_select(column_mask)!
assert selected.to_array() == [2, 3, 5, 6]

row_mask := vtl.from_array([true, false], [2, 1])!
filled := values.masked_fill(row_mask, -1)!
assert filled.to_array() == [-1, -1, -1, 4, 5, 6]
```
