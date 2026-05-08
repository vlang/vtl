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
// amax = [1, 1]  → row 0: max is at col 1 (5.0), row 1: max is at col 1 (4.0)

amin := t.argmin_axis[f64](0)!
// amin = [1, 0]  → col 0: min is at row 1 (1.0), col 1: min is at row 0 (4.0)

// Global: index of the largest element (no axis)
global_max := t.argmax[int](0)!
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
