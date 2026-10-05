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
determinant := la.det[f64](a)!
```

Available operations include `dot`, `matmul`, `det`, `inv`, `trace`, `norm`,
`outer`, `cross`, `solve`, `lstsq`, `qr`, `lu`, `cholesky`, `pinv`, and
`matrix_rank`. Matrix multiplication and decomposition requirements (rank,
shape, and tolerances) are checked by the individual functions. More examples:
[linear algebra tutorial](../docs/TUTORIAL_LINEAR_ALGEBRA.md) and
[advanced tutorial](../docs/TUTORIAL_ADVANCED_LA.md).
