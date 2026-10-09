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

`matmul` also follows NumPy's vector promotion rules: vector-vector returns a
scalar tensor, matrix-vector and vector-matrix return vectors, and vector
operands broadcast across batches of matrices. Scalar operands are rejected.

`la.multi_dot` chooses a matrix-chain parenthesization that minimizes the
estimated scalar multiplication count. A vector may appear only at the first
or last position; intermediate operands must be 2-D matrices. Batched matrices
are not accepted, matching NumPy's `linalg.multi_dot` input contract.

```v
import vtl
import vtl.la

a := vtl.ones[f64]([100, 10])
b := vtl.ones[f64]([10, 1000])
c := vtl.ones[f64]([1000, 1])
result := la.multi_dot[f64]([a, b, c])!
assert result.shape == [100, 1]
assert result.get_nth(0) == 10000.0
```

`la.solve(A, B)` follows [NumPy 2.0 right-hand-side shape rules](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html):
only a 1-D `B` is a vector. A 2-D `B` is a matrix of right-hand sides,
including when `A` has leading batch dimensions. Leading batches of `A` and
matrix `B` broadcast.

```v
import vtl
import vtl.la

u := vtl.from_1d([1.0, 2.0])!
m := vtl.from_2d([[3.0, 4.0], [5.0, 6.0]])!
dot_result := la.matmul(u, u)! // scalar tensor: 5
matrix_vector := la.matmul(m, u)! // shape [2]: [11, 17]
vector_matrix := la.matmul(u, m)! // shape [2]: [13, 16]
```

## Kronecker product

`kron(a, b)` builds the block-wise Kronecker product. It supports tensors of
any rank; the lower-rank shape is left-padded with ones before multiplying
aligned dimensions, matching NumPy's rank behavior. Inputs may be views, and
the result is a new row-major tensor.

```v
import vtl

a := vtl.from_2d([[1, 2], [3, 4]])!
b := vtl.from_2d([[0, 5], [6, 7]])!
product := vtl.kron(a, b)!
println(product.shape) // [4, 4]
println(product)
// [[0, 5, 0, 10], [6, 7, 12, 14],
//  [0, 15, 0, 20], [18, 21, 24, 28]]
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

## Vector norms

`la.vector_norm` computes a p-norm over all elements of any-rank tensor.
`la.vector_norm_axis` computes one p-norm per slice and removes the reduced
axis, matching NumPy's default. Use `la.vector_norm_axis_keepdims` to retain
the reduced axis with length one. All three return `f64` tensors. Any finite
positive or negative p, `0`, and positive or negative infinity are supported.

```v
import math
import vtl
import vtl.la

values := vtl.from_1d([3.0, -4.0, 0.0])!
l2 := la.vector_norm(values, 2)!.get_nth(0) // 5.0
l1 := la.vector_norm(values, 1)!.get_nth(0) // 7.0
nonzero := la.vector_norm(values, 0)!.get_nth(0) // 2.0
maximum := la.vector_norm(values, math.inf(1))!.get_nth(0) // 4.0

matrix := vtl.from_2d([[3.0, 4.0], [0.0, 12.0]])!
row_lengths := la.vector_norm_axis(matrix, 2, 1)! // shape [2]
row_lengths_keepdims := la.vector_norm_axis_keepdims(matrix, 2, 1)! // shape [2, 1]
```

The implementation scales values before exponentiation to reduce overflow and
underflow for finite norms. Empty inputs return zero for order zero and finite
positive orders; negative orders and extrema orders that need a minimum or
maximum return an error.

## Batched matrix norms

`la.matrix_norm` computes one matrix norm for each trailing `M x N` matrix in
an N-D tensor. It supports Frobenius (`'fro'`), nuclear (`'nuc'`), maximum and
minimum column sums (`'1'`, `'-1'`), maximum and minimum row sums (`'inf'`,
`'-inf'`), and spectral (`'2'`, `'-2'`) orders. The default is Frobenius.
`keepdims: true` retains the final
two axes as dimensions of length one. Pass matrix orders as strings; spectral
and nuclear orders reject non-finite inputs instead of passing them to SVD.

```v
import vtl
import vtl.la

batch := vtl.from_array([3, 0, 0, 0, 4, 0, 5, 0, 0, 0, 12, 0], [2, 2, 3])!
frobenius := la.matrix_norm(batch)! // [5, 13]
nuclear := la.matrix_norm(batch, ord: 'nuc')! // [7, 17]
kept := la.matrix_norm(batch, keepdims: true)! // shape [2, 1, 1]
```

`la.trace_axes` traces selected axes in a batched tensor. Axes may be negative,
and `offset` selects an upper or lower diagonal:

```v
import vtl
import vtl.la

batch := vtl.from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 2, 3])!
traces := la.trace_axes(batch, axis1: 1, axis2: 2)! // [6, 18]
offset_traces := la.trace_axes(batch, axis1: 1, axis2: 2, offset: 1)! // [8, 20]
```

`la.svdvals` returns descending singular values for every trailing matrix in
a stack. Its result has shape `[..., min(M, N)]` and rejects non-finite input:

```v
import vtl
import vtl.la

batch := vtl.from_array([3, 0, 0, 0, 4, 0, 5, 0, 0, 0, 12, 0], [2, 2, 3])!
singular_values := la.svdvals(batch)! // [[4, 3], [12, 5]]
```

`la.svd` also returns the factors. By default, `u` and `vt` are square as in
NumPy; pass `full_matrices: false` for the smaller reduced factors:

```v
import vtl
import vtl.la

matrix := vtl.from_2d([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])!
u, singular_values, vt := la.svd(matrix, full_matrices: false)!
println('Shapes: ${u.shape}, ${singular_values.shape}, ${vt.shape}') // [3, 2], [2], [2, 2]
```

`la.slogdet` computes the sign and log absolute determinant without forming
the determinant, so very large or very small determinants do not overflow or
underflow. Singular matrices return sign `0` and log absolute determinant
`-Inf`:

```v
import vtl
import vtl.la

matrix := vtl.from_2d([[1.0, 2.0], [3.0, 4.0]])!
sign, logabsdet := la.slogdet(matrix)!
println('${sign.to_array()}, ${logabsdet.to_array()}') // [-1], [0.693...]
```

`la.matrix_power` raises every trailing square matrix to an integer exponent.
It uses exponentiation by squaring; negative powers invert each matrix first
and return an error for singular inputs:

```v
import vtl
import vtl.la

rotation := vtl.from_2d([[0.0, 1.0], [-1.0, 0.0]])!
fourth_power := la.matrix_power(rotation, 4)! // identity matrix
inverse := la.matrix_power(rotation, -1)!
```

`la.cond` defaults to the 2-norm condition number and also supports the
`-2`, `1`, `-1`, `inf`, `-inf`, and `fro` matrix orders through
`la.CondOptions`:

```v
import vtl
import vtl.la

matrix := vtl.from_2d([[1.0, 2.0], [3.0, 4.0]])!
spectral := la.cond(matrix, la.CondOptions{})!
one_norm := la.cond(matrix, la.CondOptions{ ord: '1' })!
```

`la.det` and `la.inv` accept a stack of square matrices and preserve its batch
dimensions. `la.inv` returns an error when any matrix is singular:

```v
import vtl
import vtl.la

batch := vtl.from_array([1.0, 2, 3, 4, 2, 0, 0, 4], [2, 2, 2])!
determinants := la.det(batch)!
inverses := la.inv(batch)!
```

`la.eigh` computes eigenvalues and eigenvectors for stacked real symmetric
matrices; eigenvalues are ascending and eigenvectors are columns. Use
`la.eigvalsh` when only eigenvalues are needed. `EighOptions{ uplo: 'U' }`
selects the upper triangle instead of the default lower triangle. Inputs must
be finite; non-convergence within 100 Jacobi sweeps returns an error:

```v okfmt
import vtl
import vtl.la

batch := vtl.from_array([1.0, 2, 2, 1, 4, 1, 1, 2], [2, 2, 2])!
values, vectors := la.eigh(batch, la.EighOptions{})!
println(values.to_array())
println(vectors.to_array())
```

`la.matrix_rank` uses an explicit positive absolute tolerance or a dtype-aware
default. `la.matrix_rank_batch` applies the same rule independently to every
trailing matrix and preserves the batch dimensions:

```v
import vtl
import vtl.la

batch := vtl.from_array([1.0, 0, 0, 1, 1, 1, 1, 2, 2, 4, 3, 6], [2, 3, 2])!
ranks := la.matrix_rank_batch(batch, 0)!
println(ranks.to_array()) // [2, 1]
```

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

## Einstein summation

`einsum` expresses contractions by assigning a label to each axis. Labels that
appear in both operands are contracted when omitted from the output. For
example, matrix multiplication can be written explicitly or with NumPy-style
implicit output labels:

```v
import vtl

a := vtl.from_2d([[1.0, 2.0], [3.0, 4.0]])!
b := vtl.from_2d([[5.0, 6.0], [7.0, 8.0]])!
product := vtl.einsum[f64]('ij,jk->ik', a, b)!
implicit_product := vtl.einsum[f64]('ij,jk', a, b)!
println(product) // [[19, 22], [43, 50]]
```

Repeated labels select diagonals (`'ii->i'`) or reduce them to a scalar
(`'ii->'`). Multiple operands, size-one broadcasting, integer tensors, and
ellipsis notation for batched operations are supported. Labels are single ASCII
letters. Two-dimensional floating-point matrix products use the VSL linear
algebra kernel; other expressions use the general contraction evaluator.

The general evaluator also supports `math.complex.Complex` tensors, so complex
matrix products and contractions use the same notation without conjugating
either operand:

```v
import math.complex as cmplx
import vtl

a := vtl.from_2d[cmplx.Complex]([
	[cmplx.complex(1.0, 1.0), cmplx.complex(2.0, 0.0)],
	[cmplx.complex(3.0, -1.0), cmplx.complex(4.0, 0.0)],
])!
b := vtl.from_2d[cmplx.Complex]([
	[cmplx.complex(0.0, 1.0), cmplx.complex(2.0, 0.0)],
	[cmplx.complex(1.0, 0.0), cmplx.complex(0.0, -1.0)],
])!
product := vtl.einsum[cmplx.Complex]('ij,jk->ik', a, b)!
assert product.get_nth(0) == cmplx.complex(1.0, 1.0)
```

## See also

- [First Steps](./TUTORIAL_FIRST_STEPS.md) — tensor creation and shapes
- [Broadcasting](./TUTORIAL_BROADCASTING.md) — implicit shape expansion
- [Slicing](./TUTORIAL_SLICING.md) — extracting sub-tensors
