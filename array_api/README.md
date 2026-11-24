# `vtl.array_api`

`vtl.array_api` is the explicit namespace for VTL's incrementally implemented
Python Array API surface. It currently exposes creation and reshape operations plus broadcast-aware `where` selection, with a small capability matrix
so consumers can select the namespace without
mistaking VTL-specific functions for standard operations.

The supported specification target is `2024.12`. The matrix is intentionally
conservative: an operation is listed only after its semantics have focused
regression coverage. Unsupported operations are not re-exported from this
namespace.

```v
import vtl
import vtl.array_api as xp

x := xp.arange[f32](0.0, 6.0, 1.0)!
x = xp.reshape(x, [2, 3])!
assert xp.supports('reshape')

condition := vtl.from_2d[bool]([[true], [false]])!
positive := xp.ones[f32]([2, 3])
negative := xp.zeros[f32]([2, 3])
selected := xp.where(condition, positive, negative)!
```

This is not yet a complete conformance implementation. Consumers should check
`xp.capability_matrix()` before selecting operations dynamically.
