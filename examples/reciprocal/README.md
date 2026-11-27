# Floating-Point Reciprocal

This example computes elementwise reciprocals of f64 values.

Run it from the V modules directory:

```sh
cd ~/.vmodules
v run ./vtl/examples/reciprocal/main.v
```

The tensor API preserves f32/f64 dtype and uses IEEE behavior for zero,
infinity, and NaN. Integer and complex tensors are rejected.
