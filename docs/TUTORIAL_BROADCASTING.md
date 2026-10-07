# Tutorial: Broadcasting

Broadcasting lets VTL apply an operation between two tensors of different
(but compatible) shapes by implicitly expanding the smaller tensor to match
the larger one — without copying memory.

Use `broadcast2`, `broadcast3`, or `broadcast_n` when an operation needs
explicit broadcast views. Each returns views with a shared shape and shared
storage; `broadcast_n` requires at least one tensor and reports an error when
the input shapes cannot broadcast.

## Broadcasting rules

Two shapes are compatible if, for every dimension (aligned from the right),
the sizes are equal **or** one of them is 1.

```
Shape A: [3, 4]       compatible with [1, 4] and [4] and [3, 1]
Shape A: [3, 4]  NOT  compatible with [2, 4] or [3, 3]
```

Zero-sized dimensions follow NumPy's rules: `[0, 3]` and `[1, 3]` broadcast to
`[0, 3]`. A dimension of zero cannot broadcast with a dimension greater than
one. `broadcast_to` also rejects negative dimensions and target shapes with a
lower rank than the source instead of attempting an invalid view.

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

## Conditional selection

`vtl.where(condition, x, y)` selects values from `x` where the boolean
condition is true and from `y` otherwise. The condition and both value tensors
are broadcast to a common output shape.

## Approximate comparisons

`array_equal` checks that two tensors have identical shapes and exactly equal
elements. Floating-point values are compared with the language's exact `==`
semantics: NaNs are unequal (including to themselves), infinities of the same
sign are equal, and positive and negative zero are equal. Use it when exact
identity is intended.

`isclose` applies the NumPy tolerance rule elementwise and returns a boolean
tensor. `allclose` returns whether every broadcasted pair is close. Both take
explicit relative and absolute tolerances; NaNs compare false unless
`equal_nan` is enabled. Equal positive or negative infinities compare true.

```v
import vtl

condition := vtl.from_1d([true, false, true])!
x := vtl.from_1d([1, 2, 3])!
y := vtl.from_1d([10, 20, 30])!
selected := vtl.where(condition, x, y)!
println(selected) // [1, 20, 3]

measured := vtl.from_1d([1.0, 100.0005])!
expected := vtl.from_1d([1.0, 100.0])!
mask := measured.isclose(expected, rtol: 1e-5, atol: 1e-8)!
println(mask) // [true, true]
println(measured.allclose(expected, rtol: 1e-5, atol: 1e-8)!) // true
```

`where` reports an error when any of the three shapes cannot broadcast. The
comparison is `abs(a - b) <= atol + rtol * abs(b)`, so the second tensor
provides the relative scale, matching NumPy's argument order.
The defaults are `rtol: 1e-5` and `atol: 1e-8`; override them with named
options.

## Common pitfalls

Use `clip(min_value, max_value)` to bound every element. Bounds are scalar,
inclusive, use the tensor element type, and the result preserves that type.

```v
import vtl

values := vtl.from_1d([-2.0, 0.5, 8.0])!
println(values.clip(0.0, 1.0)!) // [0.0, 0.5, 1.0]
```

| Mistake | Fix |
|---------|-----|
| Adding `[n]` to `[m, n]` when you mean column-wise | Reshape to `[n, 1]` first |
| Forgetting that `[n]` broadcasts as the last axis | Use `reshape([1, n])` to be explicit |
| Expecting broadcasting to copy data | Broadcasting is lazy — no copy is made |

## See also

- [First Steps](./TUTORIAL_FIRST_STEPS.md) — tensor creation and properties
- [Slicing](./TUTORIAL_SLICING.md) — extracting sub-tensors
- [Map and Reduce](./TUTORIAL_MAP_REDUCE.md) — element-wise and reduction operations
