# Advanced Linear Algebra

Beyond the basic operations in [TUTORIAL_LINEAR_ALGEBRA.md](./TUTORIAL_LINEAR_ALGEBRA.md),
VTL provides several advanced linear algebra routines through the `vtl.la` module.

All functions below accept `vtl.Tensor[T]` inputs and return VTL tensors
(unlike VSL which works with its own `Matrix` type).

## Trace

`la.trace(t)` returns the sum of the diagonal elements of a square matrix.

```v
import vtl.la
import vtl

a := vtl.from_array[f64]([1.0, 2.0, 3.0, 4.0], [2, 2])!
// a = [[1, 2],
//      [3, 4]]
la.trace(a)!
// = 5.0  (1 + 4)
```

## Matrix Norm

`la.norm(t, ord)` computes a matrix norm:

| `ord` | Norm type |
|-------|-----------|
| `""` or `"F"` | Frobenius (default) — `sqrt(sum of squares)` |
| `"I"` | Infinity — max absolute row sum |
| `"1"` | 1-norm — max absolute column sum |

```v
import vtl.la
import vtl

a := vtl.from_array[f64]([3.0, 4.0, 0.0, 0.0], [2, 2])!

la.norm(a, 'F')! // 5.0
la.norm(a, 'I')! // 7.0  (max row sum: 3+4=7, 0+0=0)
la.norm(a, '1')! // 4.0  (max col sum: 3+0=3, 4+0=4)
```

## Outer Product

`la.outer(u, v)` computes the outer product `result[i,j] = u[i] * v[j]`,
producing a `u.len × v.len` matrix.

```v
import vtl.la
import vtl

u := vtl.from_1d[f64]([1.0, 2.0, 3.0])!
v := vtl.from_1d[f64]([4.0, 5.0])!
la.outer(u, v)!
// result = [[4,5],[8,10],[12,15]]
```

## Cross Product

`la.cross(u, v)` computes the 3-D cross product `u × v`.
Both inputs must have length 3.

```v
import vtl.la
import vtl

x := vtl.from_1d[f64]([1.0, 0.0, 0.0])!
y := vtl.from_1d[f64]([0.0, 1.0, 0.0])!
la.cross(x, y)!
// = [0.0, 0.0, 1.0]
```

## QR Factorisation

`la.qr(a)` decomposes an `m × n` matrix into `Q · R` where:
- `Q` is orthonormal (`m × min(m,n)`)
- `R` is upper triangular (`min(m,n) × n`)

```v
import vtl.la
import vtl

a := vtl.from_array[f64]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2])!
q, r := la.qr(a)!
// q.m == 3, q.n == 2  (orthonormal)
// r.m == 2, r.n == 2  (upper triangular)
```

## LU Factorisation

`la.lu(a)` returns the LU decomposition with partial pivoting: `P·A = L·U`.

Returns `(L, U, piv)`:
- `L` — lower triangular, unit diagonal (`m × min(m,n)`)
- `U` — upper triangular (`min(m,n) × n`)
- `piv` — zero-based pivot row indices

```v
import vtl.la
import vtl

a := vtl.from_array[f64]([2.0, 1.0, 1.0, 4.0, 3.0, 3.0, 8.0, 7.0, 9.0], [3, 3])!
l, u, piv := la.lu(a)!
```

## Solving Linear Systems

`la.solve(a, b)` solves `A · x = B` for `x` using LU decomposition.
`a` must be square and non-singular.

```v
import vtl.la
import vtl

a := vtl.from_array[f64]([2.0, 1.0, 1.0, 4.0, 3.0, 3.0, 8.0, 7.0, 9.0], [3, 3])!
b := vtl.from_array[f64]([1.0, 2.0, 3.0], [3, 1])!
la.solve(a, b)! // solution vector
```

## Least Squares

`la.lstsq(a, b)` solves the overdetermined linear least-squares problem:
`min ||A·x − B||₂` using SVD-based pseudo-inverse.

Returns `(x, residuals, rank, singular_values)`.

```v
import vtl.la
import vtl

// y = c0 + c1*x  fitted through (1,1), (2,3), (3,5)
a := vtl.from_array[f64]([1.0, 1.0, 1.0, 2.0, 1.0, 3.0], [3, 2])!
b := vtl.from_array[f64]([1.0, 3.0, 5.0], [3, 1])!
x, residuals, rank, sv := la.lstsq(a, b)!
// x ≈ [[1.0], [2.0]]  →  y = 1 + 2·x
```

## Cholesky Decomposition

`la.cholesky(a)` computes the Cholesky factorisation `A = L · Lᵀ`
for symmetric positive-definite matrices using VSL's LAPACK backend.
Returns the lower-triangular `L`.

**Note:** this example may fail on environments without full LAPACK bindings.
When unavailable, use `la.pinv` / `la.svd` paths instead.

```v
import vtl.la
import vtl

// Fallback example that works without LAPACK-specific symbols:
a := vtl.from_array[f64]([4.0, 2.0, 2.0, 3.0], [2, 2])!
la.pinv(a, 1e-10)!
```

## Pseudo-inverse

`la.pinv(a, tol)` computes the Moore-Penrose pseudo-inverse of `A` using SVD.
Singular values below `tol` are treated as zero.

```v
import vtl.la
import vtl

a := vtl.from_array[f64]([1.0, 2.0, 3.0, 4.0], [2, 2])!
la.pinv(a, 1e-10)!
```

## Matrix Rank

`la.matrix_rank(a, tol)` returns the effective numerical rank of `A`
using singular values greater than `tol`.

```v
import vtl.la
import vtl

a := vtl.from_array[f64]([1.0, 0, 0, 0, 1.0, 0, 0, 0, 0], [3, 3])!
la.matrix_rank(a, 1e-10)!
// = 2  (one singular value is zero)
```

## Combining with Autograd

All LA functions in `vtl.la` work seamlessly with `autograd.Variable`.
Simply pass the tensor inside a variable through any LA operation:

```v
import vtl
import vtl.autograd as ag

mut ctx := ag.ctx[f64]()
x := ctx.variable(vtl.from_1d[f64]([1.0, 2.0, 3.0])!)
w := ctx.variable(vtl.from_1d[f64]([0.5, 0.5, 0.5])!)

y := x.matmul(w)!
// Backward will propagate gradients through matmul automatically
```
