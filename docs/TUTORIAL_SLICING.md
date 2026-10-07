# Tutorial: Slicing

VTL supports slicing. It allows for selecting dimension subsets, whole dimension,
stepping (one out of 2 rows), reversing dimensions, counting from the end.

## Pad tensor boundaries

`vtl.pad` adds a before/after width for each axis. Choose constant fill,
edge replication, wrapping, reflection without repeating the edge, or symmetric
reflection with the edge included:

```v
import vtl

signal := vtl.from_1d([1, 2, 3])!
reflected := vtl.pad[int](signal, [[2, 2]], .reflect, 0)!
wrapped := vtl.pad[int](signal, [[1, 1]], .wrap, 0)!
assert reflected.to_array() == [3, 2, 1, 2, 3, 2, 1]
assert wrapped.to_array() == [3, 1, 2, 3, 1]
```

Widths must contain one non-negative `[before, after]` pair per dimension.
Non-constant modes require non-empty input dimensions whenever the padded
output contains elements.

```v
import math
import vtl

const xs = [1, 2, 3, 4, 5]

const ys = [1, 2, 3, 4, 5]

mut vandermont := [][]int{}

for i, x in xs {
	row := []int{}
	vandermont << row
	for y in ys {
		vandermont[i] << int(math.pow(x, y))
	}
}

t := vtl.from_2d(vandermont)!

println(t)
// [[   1,    1,    1,    1,    1],
// [   2,    4,    8,   16,   32],
// [   3,    9,   27,   81,  243],
// [   4,   16,   64,  256, 1024],
// [   5,   25,  125,  625, 3125]]

println(t.shape) // [5, 5]

println('slice: ')
slice1 := t.slice_hilo([1, 3], [3, 5])!

println(slice1)
// [[16, 32], [81, 243]]

println(slice1.shape) // [2, 2]

slice2 := t.slice_hilo([3], []int{})!

println('span slice: ')
println(slice2)
// [[   4,   16,   64,  256, 1024],
// [   5,   25,  125,  625, 3125]]

println(slice2.shape) // [2, 5]

slice3 := t.slice_hilo([]int{}, [-2])!

println('slice until: ')
println(slice3)
// [[  1,   1,   1,   1,   1],
// [  2,   4,   8,  16,  32],
// [  3,   9,  27,  81, 243]]

println(slice3.shape) // [3, 5]
```

## Moving axes

Use `moveaxis` to move one or more dimensions while preserving the order of
the others. `rollaxis` moves one dimension before a chosen position. Both
return tensor views, so the data is not copied.

```v
import vtl

video := vtl.from_array([]f32{len: 2 * 3 * 4, init: f32(index)}, [2, 3, 4])!
batch_last := video.moveaxis([0], [-1])!
println(batch_last.shape) // [3, 4, 2]

frames_first := video.rollaxis(2, 0)!
println(frames_first.shape) // [4, 2, 3]
```

Axes can be negative, counting from the end. For example,
`moveaxis([0], [-1])` moves the first axis to the last position. The source
and destination lists must have the same length and contain unique axes.

## Gathering values with `take`

`take` copies selected positions along an axis. Indices may be repeated or
negative, and the selected axis keeps its position in the output shape.

```v
import vtl

matrix := vtl.from_array([0, 1, 2, 3, 4, 5], [2, 3])!
columns := matrix.take([2, 0], 1)!
println(columns.shape) // [2, 2]
println(columns.to_array()) // [2, 0, 5, 3]
```

An index outside the selected axis returns an error. Use `slice` when you
need a range-based view instead of a gathered copy.

## Boolean masks

`masked_select` returns a one-dimensional tensor containing values where the
same-shaped boolean mask is true, in row-major logical order. `masked_fill`
returns a copy with matching positions replaced by a scalar value.
When every output position needs its own index, use `take_along_axis` with an
integer index tensor. Its rank must match the input rank, and all dimensions
outside the selected axis must have the same size.

```v
import vtl

values := vtl.from_2d([[1, 2, 3], [4, 5, 6]])!
mask := vtl.from_2d([[true, false, true], [false, true, false]])!
selected := values.masked_select(mask)!
filled := values.masked_fill(mask, 0)!
println(selected.to_array()) // [1, 3, 5]
println(filled.to_array()) // [0, 2, 0, 4, 0, 6]
```

The mask shape must match the tensor shape exactly. These methods also respect
logical iteration order for transposed and sliced views.

```v
import vtl

matrix := vtl.from_array([0, 1, 2, 3, 4, 5], [2, 3])!
indices := vtl.from_array([2, 0, 1, -1], [2, 2])!
selected := matrix.take_along_axis(indices, 1)!
println(selected.to_array()) // [2, 0, 4, 5]
```

## Slice Mutations

Slices can also be mutated with a single value, a nested sequence or array,
a tensor or tensor slice.

For certain use cases slice mutations can have less than intuitive results,
because the mutation happens on the same memory the whole time.
See the last mutation shown in the following code block for such an example
and the explanation below.

```v
import math
import vtl

const xs = [1, 2, 3, 4, 5]

const ys = [1, 2, 3, 4, 5]

mut vandermont := [][]int{}

for i, x in xs {
	row := []int{}
	vandermont << row
	for y in ys {
		vandermont[i] << int(math.pow(x, y))
	}
}

mut t := vtl.from_2d(vandermont)!

println(t)
// [[   1,    1,    1,    1,    1],
// [   2,    4,    8,   16,   32],
// [   3,    9,   27,   81,  243],
// [   4,   16,   64,  256, 1024],
// [   5,   25,  125,  625, 3125]]

println(t.shape) // [5, 5]

println('slice: ')
mut slice1 := t.slice_hilo([1, 3], [3, 5])!

println(slice1)
// [[16, 32], [81, 243]]

println(slice1.shape) // [2, 2]

t999 := vtl.tensor(999, [2, 2])

slice1.assign(t999)!

println('assign: ')
println(t)
// [[   1,    1,    1,    1,    1],
// [   2,    4,    8,  999,  999],
// [   3,    9,   27,  999,  999],
// [   4,   16,   64,  256, 1024],
// [   5,   25,  125,  625, 3125]]
```
