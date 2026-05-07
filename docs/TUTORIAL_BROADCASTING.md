# Tutorial: Broadcasting

Broadcasting lets VTL apply an operation between two tensors of different
(but compatible) shapes by implicitly expanding the smaller tensor to match
the larger one — without copying memory.

## Broadcasting rules

Two shapes are compatible if, for every dimension (aligned from the right),
the sizes are equal **or** one of them is 1.

```
Shape A: [3, 4]       compatible with [1, 4] and [4] and [3, 1]
Shape A: [3, 4]  NOT  compatible with [2, 4] or [3, 3]
```

## Element-wise operations with broadcasting

```v
import vtl

// Add a row vector [10, 20, 30] to each row of a 3×3 matrix.
a := vtl.from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])!
b := vtl.from_1d([10, 20, 30])! // shape [3], broadcasts as [1, 3]

c := a.add(b)! // shape [3, 3]
println(c)
// [[11, 22, 33],
//  [14, 25, 36],
//  [17, 28, 39]]
```

## Scalar broadcasting

A scalar (rank-0 tensor or a 1-element tensor) broadcasts to any shape:

```v
import vtl

t := vtl.from_2d([[1.0, 2.0], [3.0, 4.0]])!
s := vtl.tensor(2.0, [1]) // scalar 2.0

result := t.multiply(s)!
println(result)
// [[2.0, 4.0],
//  [6.0, 8.0]]
```

## Column-vector broadcasting

Broadcast a column vector `[n, 1]` across `n` columns:

```v
import vtl

col := vtl.from_array([1.0, 2.0, 3.0], [3, 1])! // shape [3, 1]
row := vtl.from_array([10.0, 20.0, 30.0], [1, 3])! // shape [1, 3]

outer := col.multiply(row)! // shape [3, 3] — outer product
println(outer)
// [[ 10,  20,  30],
//  [ 20,  40,  60],
//  [ 30,  60,  90]]
```

## Broadcasting in neural networks

Broadcasting is used internally by VTL's neural network layers.
For example, adding a bias vector `[out_features]` to a batch of activations
`[batch_size, out_features]` uses broadcasting automatically.

## Common pitfalls

| Mistake | Fix |
|---------|-----|
| Adding `[n]` to `[m, n]` when you mean column-wise | Reshape to `[n, 1]` first |
| Forgetting that `[n]` broadcasts as the last axis | Use `reshape([1, n])` to be explicit |
| Expecting broadcasting to copy data | Broadcasting is lazy — no copy is made |

## See also

- [First Steps](./TUTORIAL_FIRST_STEPS.md) — tensor creation and properties
- [Slicing](./TUTORIAL_SLICING.md) — extracting sub-tensors
- [Map and Reduce](./TUTORIAL_MAP_REDUCE.md) — element-wise and reduction operations