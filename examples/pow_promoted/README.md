# Promoted Tensor Power

`pow_promoted` broadcasts tensor operands and validates its explicit result
type against `promote_types`. Integer results preserve exact-width wrapping
arithmetic and reject negative integer exponents. Complex inputs and 64-bit
integers combined with floating output are unsupported to avoid lossy casts.

Run from the V modules directory:

```sh
v run ./vtl/examples/pow_promoted/main.v
```

See the [math ufunc tutorial](../../docs/TUTORIAL_MATH_UFUNCS.md).
