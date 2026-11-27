# Normalized Sinc

This example evaluates `sin(pi*x)/(pi*x)` for an f64 tensor, including the
removable singularity at zero.

Run it from the V modules directory:

```sh
cd ~/.vmodules
v run ./vtl/examples/sinc/main.v
```

See the [math ufunc tutorial](../../docs/TUTORIAL_MATH_UFUNCS.md#normalized-sinc)
for special-value behavior and dtype support.
