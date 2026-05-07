# vtl_vandermont

Builds a 5×5 Vandermonde matrix, then demonstrates tensor slicing and in-place
assignment.

## What it demonstrates

- Creating a 2-D tensor from a nested V array with `vtl.from_2d`
- Slicing a sub-tensor by row/column range with `.slice_hilo()`
- Inspecting tensor shape with `.shape`
- Assigning into a slice with `.assign()` (modifies the underlying tensor)

## How to run

```sh
v run main.v
```

## Expected output

```
[[   1,    1,    1,    1,    1],
 [   2,    4,    8,   16,   32],
 [   3,    9,   27,   81,  243],
 [   4,   16,   64,  256, 1024],
 [   5,   25,  125,  625, 3125]]
[5, 5]
slice:
[[ 27,   81],
 [ 64,  256]]
[2, 2]
span slice:
[[ 4,   16,   64,  256, 1024],
 [ 5,   25,  125,  625, 3125]]
[2, 5]
slice until:
[[1, 1, 1, 1, 1],
 [2, 4, 8, 16, 32],
 [3, 9, 27, 81, 243]]
[3, 5]
assign:
[[   1,    1,    1,    1,    1],
 [   2,    4,    8,   16,   32],
 [   3,    9,  999,  999,  243],
 [   4,   16,  999,  999, 1024],
 [   5,   25,  125,  625, 3125]]
```

## Key API

| Function / method | Description |
|-------------------|-------------|
| `vtl.from_2d(arr)` | Create a 2-D tensor from a nested V array |
| `t.slice_hilo(rows, cols)` | Slice by [start, end) row and column ranges |
| `t.shape` | `[]int` with the size of each dimension |
| `slice.assign(t)` | Write values from `t` into the slice's memory |

## Related examples

- [`vtl_basic_usage`](../vtl_basic_usage/) — 1-D tensors and basic operations
- [`autograd_backprop`](../autograd_backprop/) — automatic differentiation
