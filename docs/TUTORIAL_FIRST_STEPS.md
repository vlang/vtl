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

t := vtl.from_2d([[1, 2, 3], [4, 5, 6]])

println(t)
// [[1, 2, 3], [4, 5, 6]]

println(t.size) // 6
println(t.rank()) // 2
println(t.shape) // [2, 3]
println(t.strides) // [3, 1] => next row is 3 elements away in memory while the next column is 1 element away in memory
```

## Tensor Creation

The canonical way to create a tensor is to use the `vtl.from_*` functions.

```v
import vtl

t1d := vtl.from_1d([1, 2, 3])

println(t1d)
// [1, 2, 3]

t2d := vtl.from_2d([[1, 2, 3], [4, 5, 6]])

println(t2d)
// [[1, 2, 3], [4, 5, 6]]

t := vtl.from_array([1, 2, 3, 4, 5, 6, 7, 8], [2, 4])

println(t)
// [[1, 2, 3, 4], [5, 6, 7, 8]]

println(t.size) // 8
println(t.rank()) // 2
println(t.shape) // [2, 4]
println(t.strides) // [4, 1] => next row is 4 elements away in memory while the next column is 1 element away in memory
```

Other ways to create a tensor are:

- `new_tensor` - creates a new tensor.
  This can be used to initialize a tensor of a specific shape with a default value.
  (0 for numbers, false for bool, ...)
- `zeros` - creates a tensor of zeros.
- `ones` - creates a tensor of ones.
- `zeros_like` - creates a tensor of zeros with the same shape as the given tensor.
- `ones_like` - creates a tensor of ones with the same shape as the given tensor.

```v
import vtl

t := vtl.new_tensor(0.0, [2, 3])

println(t)
// [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

booleans := vtl.new_tensor(false, [2, 3])

println(booleans)
// [[false, false, false], [false, false, false]]

z := vtl.zeros<f64>([2, 3])

println(z)
// [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

o := vtl.ones<f64>([2, 3])

println(o)
// [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]

tmp := vtl.from_array([1, 2, 3, 4], [2, 2])

h := vtl.zeros_like(tmp)

println(h)
// [[0.0, 0.0], [0.0, 0.0]]

i := vtl.ones_like(tmp)

println(i)
// [[1.0, 1.0], [1.0, 1.0]]
```

## Accessing and modifying a value

```v
import vtl

mut t := vtl.from_array([1, 2, 3, 4, 5, 6, 7, 8], [2, 4])

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

a := vtl.from_array([1, 2, 3, 4, 5, 6, 7, 8], [2, 4])

println(a)
// [[1, 2, 3, 4], [5, 6, 7, 8]]

b := a.reshape([4, 2])

println(b)
// [[1, 2], [3, 4], [5, 6], [7, 8]]
```

Here modifying `b` WILL modify `a`. This behaviour is the same as Numpy and Julia.
