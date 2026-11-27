# Promoted Tensor Power

`pow_promoted` broadcasts tensor operands and validates its explicit result
type against `promote_types`. The example also shows `binary_promoted_auto`
and `scalar_left_promoted_auto` selecting result dtypes at runtime; these return
a `PromotedTensor` sum type that callers inspect or match. Integer results
preserve exact-width wrapping arithmetic and reject negative integer exponents.
64-bit integers combined with floating output are rejected to avoid lossy
casts, while complex floor division, remainder, and fmod remain unsupported.

Run from the V modules directory:

```sh
v run ./vtl/examples/pow_promoted/main.v
```

See the [math ufunc tutorial](../../docs/TUTORIAL_MATH_UFUNCS.md).
