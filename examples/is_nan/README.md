# Elementwise NaN detection

Use `Tensor.is_nan()`, `Tensor.is_inf(sign)`, and `Tensor.is_finite()` to build
boolean masks for NaN, positive/negative infinity, and finite values. Each
operation preserves the input shape and works with tensor views.

Run from `~/.vmodules`:

```sh
v run ./vtl/examples/is_nan/main.v
```

The example prints all three masks for a tensor containing NaN, finite values,
and both infinities.
