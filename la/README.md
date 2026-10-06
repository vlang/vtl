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
```

Available operations include `dot`, `matmul`, `tensordot`, `det`, `inv`,
`trace`, matrix `norm`, vector `vector_norm`, axis `vector_norm_axis` and
`vector_norm_axis_keepdims`,
`outer`, `cross`, `solve`, `lstsq`, `qr`, `lu`, `cholesky`, `pinv`, and
`matrix_rank`. Matrix multiplication and decomposition requirements (rank,
shape, and tolerances) are checked by the individual functions.

`tensordot(a, b, axes)` contracts the last `axes` dimensions of `a` with the
first `axes` dimensions of `b`. Use `tensordot_axes(a, b, axes_a, axes_b)` to
choose arbitrary axis pairs.

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
