# Elementwise NaN detection

Use `Tensor.is_nan()` to build a boolean tensor that marks NaN elements while
leaving finite values and positive or negative infinity unmarked. The operation
preserves the input shape and works with tensor views.

Run from `~/.vmodules`:

```sh
v run ./vtl/examples/is_nan/main.v
```

The example prints `[true, false, false, false]` for NaN, a finite value, and
both infinities.
