# Indexing, gather, and scatter

`take` and `take_along_axis` gather copies from a tensor. Negative axes and
indices count from the end. `take_along_axis` broadcasts dimensions outside
the selected axis. See [slicing](./TUTORIAL_SLICING.md) when a view of a range
is more appropriate.

```v
import vtl

matrix := vtl.from_2d([[10, 11, 12], [20, 21, 22]])!
indices := vtl.from_array[int]([2, 0], [1, 2])!
selected := matrix.take_along_axis(indices, 1)!
assert selected.shape == [2, 2]
assert selected.to_array() == [12, 10, 22, 20]
```

## Gathering with autograd

`Variable.take_along_axis` records the gather in the computation graph. During
backpropagation, gradients are scattered to the selected source positions;
repeated indices accumulate, and broadcast source dimensions sum their
contributions. Integer indices are treated as constants and snapshotted when
the gather is recorded, so later edits to the index tensor do not change the
backward path.

```v ignore
import vtl
import vtl.autograd

mut ctx := autograd.ctx[f64]()
input := ctx.variable(vtl.from_array([10.0, 20.0, 30.0], [1, 3])!)
indices := vtl.from_array[int]([1, 1, 2], [1, 3])!
mut selected := input.take_along_axis(indices, 1)!
selected.backprop()!
println(input.grad.to_array()) // [0.0, 2.0, 1.0]
```

See the runnable [autograd gather example](../examples/autograd_gather/README.md)
for repeated indices and the resulting gradient.

`Variable.scatter_add` returns an updated copy and records gradients for both
the source and update tensors. The source receives the output gradient at each
position; each update receives the gradient at its destination. Duplicate
destinations accumulate in the forward pass while each update keeps its own
gradient.

See the runnable [autograd scatter example](../examples/autograd_scatter/README.md)
for an indexed update with a non-uniform downstream gradient.

`Variable.put_along_axis` follows replacement semantics. The source gradient is
zero at overwritten positions, and when an index is repeated only the last
update receives the destination gradient. Earlier updates to that same
position receive zero.

See the runnable [autograd put example](../examples/autograd_put/README.md) for
duplicate destinations and their gradients.

## Count non-zero values

`count_nonzero` counts all non-zero tensor values. `count_nonzero_axis` counts
along one axis and removes it; pass `keepdims: true` to retain that axis with
length one.

```v
import vtl

values := vtl.from_2d([[0, 2, 0], [3, 0, 4]])!
assert vtl.count_nonzero[int](values) == 3
assert vtl.count_nonzero_axis[int](values, 1, false)!.to_array() == [1, 2]
assert vtl.count_nonzero_axis[int](values, 1, true)!.shape == [2, 1]
```

## Search and digitize

`searchsorted` returns insertion positions in an ascending one-dimensional
tensor; `searchsorted_descending` handles descending input. Both assume that
the input is already sorted and use binary search for each query. Choose
`.left` to insert before equal values or `.right` to insert after them. The
result has the shape of the query tensor.
`digitize` assigns values to monotonic bin edges, including decreasing edges;
`right` controls which interval owns an edge value.

```v
import vtl

sorted := vtl.from_1d([1, 3, 3, 5])!
queries := vtl.from_1d([0, 3, 4])!
assert vtl.searchsorted(sorted, queries, .left)!.to_array() == [0, 1, 3]
assert vtl.searchsorted(sorted, queries, .right)!.to_array() == [0, 3, 3]

descending := vtl.from_1d([9, 7, 7, 4, 1])!
assert vtl.searchsorted_descending(descending, queries, .left)!.to_array() == [5, 4, 3]

measurements := vtl.from_1d([0.2, 1.5, 2.0, 3.8, 5.1])!
bins := vtl.from_1d([0.0, 2.0, 4.0, 6.0])!
assert vtl.digitize(measurements, bins, false)!.to_array() == [1, 1, 2, 2, 3]
```

Passing an unsorted tensor to either search function violates its precondition.
`digitize` checks its bins and rejects non-monotonic edges.

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
import vtl

values := vtl.from_2d([[0, 2, 0], [3, 4, 0]])!
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

`put` updates row-major logical flat positions and repeats a shorter values
tensor as needed. Negative indices count from the end in the default `.raise`
mode. `put_with_mode` also accepts `.wrap` and `.clip`; `.clip` maps negative
indices to the first position, matching NumPy. Invalid indices are rejected
before any writes, including when the target is a non-contiguous view.

```v
import vtl

mut values := vtl.from_2d([[10, 11, 12], [20, 21, 22]])!
indices := vtl.from_1d([0, -1, 0])!
updates := vtl.from_1d([90, 99])!
values.put(indices, updates)!
assert values.to_array() == [90, 11, 12, 20, 21, 99]

mut wrapped := vtl.from_1d([0, 0, 0, 0])!
wrapped.put_with_mode(vtl.from_1d([-1, 4])!, vtl.from_1d([7, 8])!, .wrap)!
assert wrapped.to_array() == [8, 0, 0, 7]
```

Boolean masks passed to `masked_select` and `masked_fill` may broadcast to the
tensor shape. `masked_select` returns selected elements as a one-dimensional
copy; `masked_fill` returns a new tensor and leaves the input unchanged. For
NumPy-style `array[mask]`, `boolean_index` requires the mask to match the
leading tensor dimensions and retains any trailing dimensions.

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

```v
import vtl

cube := vtl.from_array([]int{len: 12, init: index}, [2, 3, 2])!
mask := vtl.from_2d([[true, false, true], [false, true, false]])!
selected_cube := vtl.boolean_index[int](cube, mask)!
assert selected_cube.shape == [3, 2]
assert selected_cube.to_array() == [0, 1, 4, 5, 8, 9]
```

## Mixed basic and coordinate-array indexing

`mixed_index` accepts one descriptor per indexed axis; omitted trailing axes
select the full axis. Use `integer_index` to remove one axis, `slice_index` for
an explicit range, `slice_all` for a full or reversed range, and `array_index`
for integer coordinate tensors. Coordinate tensors broadcast together. When
their axes are separated by slices, the broadcast dimensions move to the front
as in NumPy. Coordinate indexing returns an independent copy; basic indexing
returns a view except for negative-step slices, which currently materialize a
copy because VTL storage cannot represent negative-stride views safely.

```v
import vtl

matrix := vtl.from_array[int]([]int{len: 35, init: index}, [5, 7])!
rows := vtl.from_1d([0, 2, 4])!
columns := vtl.slice_index(1, 3, 1)!
selected := vtl.mixed_index[int](matrix, [vtl.array_index(rows), columns])!
assert selected.shape == [3, 2]
assert selected.to_array() == [1, 2, 15, 16, 29, 30]

mut last_row := vtl.mixed_index[int](matrix, [vtl.integer_index(-1)])!
last_row.set([0], 99)
assert matrix.get([4, 0]) == 99 // basic indexing shares storage
```

Use `ellipsis_index` to select all remaining axes at that position, and
`newaxis_index` to insert a size-one axis without consuming an input axis.
New axes participate in advanced-index placement as basic dimensions; basic
new-axis selections remain views. Multiple ellipses and excess consuming
indices return errors.

```v
import vtl

volume := vtl.from_array[int]([]int{len: 24, init: index}, [2, 3, 4])!
last_columns := vtl.mixed_index[int](volume, [vtl.ellipsis_index(), vtl.slice_index(1, 4, 2)!])!
assert last_columns.shape == [2, 3, 2]

row := vtl.from_array[int]([1], [1])!
expanded := vtl.mixed_index[int](volume, [vtl.array_index(row), vtl.newaxis_index(),
	vtl.ellipsis_index()])!
assert expanded.shape == [1, 1, 3, 4]
```

Use `advanced_index` when each source axis has a coordinate tensor:

## Coordinate-array indexing

`advanced_index` accepts one integer coordinate tensor for each input axis.
Those coordinate tensors broadcast together, and each matching coordinate
tuple selects one value. Negative coordinates count from the end. The result
is an independent row-major copy, including when the source is a view.

```v
import vtl

matrix := vtl.from_2d([[10, 11, 12], [20, 21, 22]])!
rows := vtl.from_array[int]([0, 1], [2, 1])!
columns := vtl.from_1d([1, 2, 0])!
selected := vtl.advanced_index[int](matrix, [rows, columns])!
assert selected.shape == [2, 3]
assert selected.to_array() == [11, 12, 10, 21, 22, 20]
```

This covers coordinate-array selection such as NumPy's `matrix[rows, columns]`.
Use `mixed_index` when coordinate arrays are combined with scalar or range
indices. Use `take` or `take_nd` when gathering along one axis.
