# Replace non-finite values

`Tensor.nan_to_num` replaces NaN and positive or negative infinity in f32/f64
tensors. By default it uses zero for NaN and the type's largest finite values
for infinities; named options override each replacement.

## Run

Run from `~/.vmodules`:

```sh
v run ./vtl/examples/nan_to_num/main.v
```

## Notes

Integer tensors are copied unchanged. Complex tensor support is not implemented.
