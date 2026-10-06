# Tutorial: Matrix and Vector operations

The following linear algebra operations are supported for
tensors of rank 1 (vectors) and 2 (matrices):

- dot product (Vector to Vector) using `vtl.la.dot`
- addition and subtraction (any rank) using `vtl.add` and `vtl.subtract`
- multiplication or division by a scalar using `vtl.multiply` and `vtl.divide`
- matrix-matrix multiplication using `vtl.la.matmul`
- generalized tensor contraction using `vtl.la.tensordot`
- . . .

*Note*: Matrix operations for floats are accelerated using
[vsl.blas](https://github.com/vlang/vsl/tree/master/blas).
Unfortunately there is no acceleration routine for integers.
Integer matrix-matrix and matrix-vector multiplications
are implemented via semi-optimized routines.

## Creating vectors and matrices

```v
import vtl

// 1-D vector
v := vtl.from_1d([1.0, 2.0, 3.0])!

// 2-D matrix (3 rows × 3 columns)
a := vtl.from_2d([
	[1.0, 2.0, 3.0],
	[4.0, 5.0, 6.0],
	[7.0, 8.0, 9.0],
])!
```

## Element-wise arithmetic

```v
import vtl

a := vtl.from_2d([[1.0, 2.0], [3.0, 4.0]])!
b := vtl.from_2d([[5.0, 6.0], [7.0, 8.0]])!

sum := a.add(b)! // [[6, 8], [10, 12]]
diff := a.subtract(b)! // [[-4, -4], [-4, -4]]
prod := a.multiply(b)! // element-wise: [[5, 12], [21, 32]]
quot := a.divide(b)! // element-wise: [[0.2, 0.333], [0.428, 0.5]]
```

## Scalar operations

```v
import vtl

t := vtl.from_1d([2.0, 4.0, 6.0])!
s := vtl.tensor(2.0, [1])

scaled := t.multiply(s)! // [4.0, 8.0, 12.0]
```

## Dot product (vectors)

```v
import vtl
import vtl.la

u := vtl.from_1d([1.0, 2.0, 3.0])!
v := vtl.from_1d([4.0, 5.0, 6.0])!

d := la.dot(u, v)! // 1*4 + 2*5 + 3*6 = 32.0
println(d)
```

## Matrix multiplication

```v
import vtl
import vtl.la

a := vtl.from_2d([[1.0, 2.0], [3.0, 4.0]])! // 2×2
b := vtl.from_2d([[5.0, 6.0], [7.0, 8.0]])! // 2×2

c := la.matmul(a, b)! // 2×2
println(c)
// [[19, 22],
//  [43, 50]]
```

## Tensor contraction

`tensordot` generalizes dot products and matrix multiplication by summing over
one or more matching dimensions. With an integer `axes`, it contracts the last
axes of the first tensor with the first axes of the second tensor. Use
`tensordot_axes` when the contracted axes are elsewhere:

```v
import vtl
import vtl.la

a := vtl.from_2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])!
b := vtl.from_2d([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])!
product := la.tensordot(a, b, 1)! // shape [2, 2], same contraction as matmul

// Contract axis 0 of a with axis 0 of b instead.
contracted := la.tensordot_axes(a, b, [0], [0])!
```

The result shape lists the uncontracted axes of `a`, followed by the
uncontracted axes of `b`. Axis lists must have equal lengths, contain no
repeated axes, and pair dimensions with equal sizes. Negative axes count from
the end of their tensor.

## Transpose

Pass the desired axis order to `transpose`.  For a 2-D matrix, swap axes `[1, 0]`:

```v
import vtl

a := vtl.from_2d([[1, 2, 3], [4, 5, 6]])! // shape [2, 3]
t := a.transpose([1, 0])! // shape [3, 2]
println(t)
// [[1, 4],
//  [2, 5],
//  [3, 6]]
```

## See also

- [First Steps](./TUTORIAL_FIRST_STEPS.md) — tensor creation and shapes
- [Broadcasting](./TUTORIAL_BROADCASTING.md) — implicit shape expansion
- [Slicing](./TUTORIAL_SLICING.md) — extracting sub-tensors