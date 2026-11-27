# `vtl.array_api`

`vtl.array_api` is the explicit namespace for VTL's incrementally implemented
Python Array API surface. It currently exposes creation, reshape, broadcasting,
and transpose operations plus broadcast-aware `where` selection, `matmul`, and scalar
`clip`, global `sum`, `prod`, `min`, `max`, `all`, `any_`, and `mean`, with a small capability matrix so consumers can select the namespace without
mistaking VTL-specific functions for standard operations.

The supported specification target is `2024.12`. The matrix is intentionally
conservative: an operation is listed only after its semantics have focused
regression coverage. Unsupported operations are not re-exported from this
namespace.

Numeric reductions (`sum`, `prod`, `min`, `max`, and `mean`) accept VTL's
`vtl.Real` constraint, so unsupported element types are rejected at compile
time.

```v
import vtl
import vtl.array_api as xp

x := xp.arange[f32](0.0, 6.0, 1.0)!
x = xp.reshape(x, [2, 3])!
assert xp.supports('reshape')
expanded := xp.broadcast_to(x, [4, 2, 3])!
transposed := xp.transpose(expanded, 0, 2, 1)!

condition := vtl.from_2d[bool]([[true], [false]])!
positive := xp.ones[f32]([2, 3])
negative := xp.zeros[f32]([2, 3])
selected := xp.where(condition, positive, negative)!

weights := xp.ones[f32]([3, 2])
scores := xp.matmul(weights, weights.t()!)!
bounded := xp.clip(scores, 0.0, 1.0)!
total := xp.sum(bounded)
average := xp.mean(bounded)
close_mask := xp.isclose(bounded, bounded, 1e-5, 1e-8, false)!
finite_mask := xp.isfinite(bounded)
```

This is not yet a complete conformance implementation. Consumers should check
`xp.capability_matrix()` before selecting operations dynamically.
