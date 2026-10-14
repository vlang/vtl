# Replace non-finite values

Tensor.nan_to_num replaces NaN and positive or negative infinity in f32/f64
and complex128 tensors. For complex values, each component is replaced
independently. By default it uses zero for NaN and the type's largest finite
values for infinities; named options override each replacement.
The component-wise complex behavior follows
[NumPy's nan_to_num reference](https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html).

## Run

Run from `~/.vmodules`:

```sh
v run ./vtl/examples/nan_to_num/main.v
```

## Notes

Integer tensors are copied unchanged. Variable.nan_to_num currently supports
real floating-point tensors only; complex autograd remains future work.
