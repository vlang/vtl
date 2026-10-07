# `vtl.la`

Linear algebra wrappers over VSL. Functions accept VTL tensors and return VTL
tensors; many decompositions currently return `f64` results even when the input
has another numeric type. Check each function's signature when preserving dtype
matters.

```v
import vtl
import vtl.la

a := vtl.from_2d[f64]([[2.0, 0.0], [0.0, 4.0]])!
b := vtl.from_2d[f64]([[1.0, 2.0], [3.0, 4.0]])!
c := la.matmul[f64](a, b)!
contracted := la.tensordot[f64](a, b, 1)!
determinant := la.det[f64](a)!
main_diagonal := la.diag[f64](a, 0)!
```

Available operations include `dot`, `matmul`, `tensordot`, `diag`, batched
`det` and `inv`, `trace`, matrix `norm`, vector `vector_norm` and its axis
variants, `outer`, `cross`, `solve`, `lstsq`, `qr`, `lu`, `cholesky`, `pinv`,
`matrix_rank`, `matrix_rank_batch`, `slogdet`, `svdvals`, batched `matrix_norm`,
`matrix_power`, `cond`, and symmetric `eigh`/`eigvalsh`. Shape and tolerance
requirements are checked by each function.

`solve(a, b)` accepts square matrices or stacks of square matrices. Leading
batch dimensions follow NumPy broadcasting. A vector right-hand side has shape
`[n]` (or `[..., n]` for vector batches); a matrix right-hand side has shape
`[..., n, nrhs]`. The result keeps the broadcast batch shape and removes the
last axis for vector right-hand sides. Singular systems return an error.

`lstsq(a, b)` accepts a vector or matrix right-hand side. It preserves vector
output shape, reports the effective numerical rank, returns squared residuals
for overdetermined full-column-rank systems, and an empty residual tensor for
rank-deficient or underdetermined systems.

`tensordot(a, b, axes)` contracts the last `axes` dimensions of `a` with the
first `axes` dimensions of `b`. Use `tensordot_axes(a, b, axes_a, axes_b)` to
choose arbitrary axis pairs.

`vector_norm_axes(t, ord, axes, keepdims)` reduces a tuple of axes in one
operation. It requires at least one axis; axes may be negative but must be
unique. For example, `vector_norm_axes(t, 2, [0, 2], false)` reduces the first
and last dimensions and keeps the middle dimension.

`diag(input, offset)` builds a matrix from a vector or extracts a diagonal
vector from a matrix. Positive offsets select diagonals above the main one;
negative offsets select diagonals below it. It accepts rank-1 and rank-2
inputs and returns a copy.

`covariance_matrix(data, rowvar, ddof)` and `correlation_matrix(data, rowvar)`
accept rank-2 tensors. With `rowvar: true`, each row is a variable and columns
are observations, matching NumPy's default. Set `rowvar: false` when variables
are columns. Covariance uses `ddof: 0` for population values and `ddof: 1` for
sample values. Non-positive degrees of freedom follow NumPy's divisor
semantics, including NaN when the divisor is zero. Correlation returns NaN for
constant variables.

More examples:
[linear algebra tutorial](../docs/TUTORIAL_LINEAR_ALGEBRA.md) and
[advanced tutorial](../docs/TUTORIAL_ADVANCED_LA.md).
