# Tutorial: First Steps

## Tensor Properties

Tensors have the following properties:

- `shape` - the shape of the tensor. It is a sequence of a the tensor's dimensions along each axis.
- `strides` - the strides of the tensor.
  It is a sequence of numbers of steps to get the next item along a dimension.
- `size` - the total number of elements in the tensor.
- `rank()` - the number of dimensions in the tensor.
  It is `0` for scalars, `1` for vectors, `2` for matrices and `N` for tensors of rank `N`.

```v
import vtl

t := vtl.from_2d([[1, 2, 3], [4, 5, 6]])!

println(t)
// [[1, 2, 3], [4, 5, 6]]

println(t.size) // 6
println(t.rank()) // 2
println(t.shape) // [2, 3]
println(t.strides) // [3, 1] => next row is 3 elements away in memory while the next column is 1 element away in memory
```

The optional memory format changes the storage layout, not the logical matrix
values. A column-major tensor created from nested rows keeps the same row and
column indices:

```v
import vtl

column_major := vtl.from_2d([[1, 2, 3], [4, 5, 6]], memory: .col_major)!
assert column_major.is_col_major_contiguous()
assert column_major.to_array() == [1, 2, 3, 4, 5, 6]
assert column_major.get([0, 1]) == 2
```

## Tensor Creation

The canonical way to create a tensor is to use the `vtl.from_*` functions.

```v
import vtl

t1d := vtl.from_1d([1, 2, 3])!

println(t1d)
// [1, 2, 3]

t2d := vtl.from_2d([[1, 2, 3], [4, 5, 6]])!

println(t2d)
// [[1, 2, 3], [4, 5, 6]]

t := vtl.from_array([1, 2, 3, 4, 5, 6, 7, 8], [2, 4])!

println(t)
// [[1, 2, 3, 4], [5, 6, 7, 8]]

println(t.size) // 8
println(t.rank()) // 2
println(t.shape) // [2, 4]
println(t.strides) // [4, 1] => next row is 4 elements away in memory while the next column is 1 element away in memory
```

Use `range` for unit-spaced integer bounds, `arange` for a chosen step, and
`linspace` when you need a fixed number of samples. `range` and `arange` exclude
the stop value; `linspace` includes it by default. Negative steps create a
descending `arange` sequence.

```v
import vtl

integers := vtl.range[int](2, 6)
fractions := vtl.arange[f64](0.0, 1.0, 0.25)!
countdown := vtl.arange[int](5, 0, -2)!
samples := vtl.linspace[f64](0.0, 1.0, 5)!
interior_samples := vtl.linspace[f64](0.0, 1.0, 4, endpoint: false)!
decades := vtl.logspace[f64](0.0, 4.0, 5)! // [1, 10, 100, 1000, 10000]

assert integers.to_array() == [2, 3, 4, 5]
assert fractions.to_array() == [0.0, 0.25, 0.5, 0.75]
assert countdown.to_array() == [5, 3, 1]
assert samples.to_array() == [0.0, 0.25, 0.5, 0.75, 1.0]
assert interior_samples.to_array() == [0.0, 0.25, 0.5, 0.75]
assert decades.to_array() == [1.0, 10.0, 100.0, 1000.0, 10000.0]
```

## Numeric dtype conversion

Use the explicit `as_*` methods when an operation needs a different tensor
element type. Numeric casts cover the supported integer, floating-point, and
boolean types while preserving the tensor's logical values and shape. A cast
to the existing type returns the original tensor; a conversion creates a
row-major tensor. Floating-point values converted to integers are truncated.

```v
import vtl

samples := vtl.from_array[u32]([1, 2, 3, 4], [2, 2])!
as_float := samples.as_f64()
as_small_int := samples.as_i16()
assert as_float.shape == samples.shape
assert as_float.to_array() == [1.0, 2.0, 3.0, 4.0]
assert as_small_int.to_array() == [1, 2, 3, 4]

fractional := vtl.from_1d([1.75, 2.5])!
assert fractional.as_int().to_array() == [1, 2]
```

`dtype()` reports a tensor's element type. `promote_types` determines a
common result type for two array dtypes and rejects string/numeric mixtures.
The promotion helper models NumPy's array-dtype rules using VTL's current
types. VTL arithmetic operators do not yet automatically dispatch across
different tensor types.

```v
import vtl

integers := vtl.from_1d[u16]([1, 2, 3])!
assert integers.dtype() == .uint16
assert vtl.promote_types(integers.dtype(), .float32)! == .float32
assert vtl.promote_types(.int64, .uint64)! == .float64
```

`logspace` spaces samples evenly in the exponent; its default base is 10. Set
`base` in the options to use another positive finite base.

Other ways to create a tensor are:

- `tensor` - creates a new tensor.
  This can be used to initialize a tensor of a specific shape with a default value.
  (0 for numbers, false for bool, ...)
- `zeros` - creates a tensor of zeros.
- `ones` - creates a tensor of ones.
- `zeros_like` - creates a tensor of zeros with the same shape as the given tensor.
- `ones_like` - creates a tensor of ones with the same shape as the given tensor.

```v
import vtl

t := vtl.tensor(0.0, [2, 3])

println(t)
// [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

booleans := vtl.tensor(false, [2, 3])

println(booleans)
// [[false, false, false], [false, false, false]]

z := vtl.zeros[f64]([2, 3])

println(z)
// [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

o := vtl.ones[f64]([2, 3])

println(o)
// [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]

tmp := vtl.from_array([1, 2, 3, 4], [2, 2])!

h := vtl.zeros_like(tmp)

println(h)
// [[0.0, 0.0], [0.0, 0.0]]

i := vtl.ones_like(tmp)

println(i)
// [[1.0, 1.0], [1.0, 1.0]]
```

## Coordinate grids

`meshgrid(x, y)` makes dense 2-D coordinate matrices with NumPy's default
`xy` indexing. The resulting shape is `[y.size, x.size]`.

```v
import vtl

x := vtl.from_1d([1, 2, 3])!
y := vtl.from_1d([10, 20])!
x_grid, y_grid := vtl.meshgrid(x, y)!
println(x_grid) // [[1, 2, 3], [1, 2, 3]]
println(y_grid) // [[10, 10, 10], [20, 20, 20]]
```

For three or more axes, `meshgrid_n` returns one grid per input. Use `.ij` to
preserve axis order or `.xy` to swap the first two axes like NumPy:

```v
import vtl

x := vtl.from_1d([1, 2])!
y := vtl.from_1d([10, 20, 30])!
z := vtl.from_1d([4, 5])!
grids := vtl.meshgrid_n[int]([x, y, z], .xy)!
assert grids[0].shape == [3, 2, 2]
assert grids[0].get([2, 1, 1]) == 2
assert grids[1].get([2, 1, 1]) == 30
assert grids[2].get([2, 1, 1]) == 5
```

## Accessing and modifying a value

```v
import vtl

mut t := vtl.from_array([1, 2, 3, 4, 5, 6, 7, 8], [2, 4])!

println(t.get([1, 1]))
// 5

t.set([1, 1], 10)

println(t)
// [[1, 2, 3, 4], [10, 5, 6, 7]]
```

## Copying a tensor

Warning: When you do the following, both tensors `a` and `b` will share the same data.
Full copy must be explicitly requested via the `copy()` function.

```v
import vtl

a := vtl.from_array([1, 2, 3, 4, 5, 6, 7, 8], [2, 4])!

println(a)
// [[1, 2, 3, 4], [5, 6, 7, 8]]

b := a.reshape([4, 2])!

println(b)
// [[1, 2], [3, 4], [5, 6], [7, 8]]
```

Here modifying `b` WILL modify `a`. This behaviour is the same as Numpy and Julia.
