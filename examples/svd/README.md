# Singular value decomposition

Run from `~/.vmodules`:

```sh
v run ./vtl/examples/svd/main.v
```

`la.svd` returns U, descending singular values, and V transpose. Pass
`full_matrices: false` for reduced factors, which avoid the extra columns in U
and rows in V transpose for rectangular matrices. See the
[linear algebra tutorial](../../docs/TUTORIAL_LINEAR_ALGEBRA.md) for details.
