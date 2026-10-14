# Elementwise Math and NumPy Ufuncs

VTL provides elementwise math operations on tensors, including decomposition of
floating-point values into integer/fractional parts or binary mantissas and
exponents.

## Split fractional and integral parts

`Tensor.modf()` returns `(fractional, integral)`, in the same order as
`numpy.modf`. It preserves the input shape and supports f32 and f64 tensors.

```v
import vtl

values := vtl.from_1d([1.25, -2.75, 4.0])!
fractional, integral := values.modf()!
println(fractional.to_array()) // [0.25, -0.75, 0.0]
println(integral.to_array()) // [1.0, -2.0, 4.0]
```

## Decompose and reconstruct binary values

`Tensor.frexp()` returns a floating-point mantissa and an integer exponent for
each value. `vtl.ldexp(mantissa, exponent)` scales the mantissas by the
corresponding powers of two and broadcasts the two inputs, so it can reconstruct
the original tensor:

```v
import vtl

values := vtl.from_1d[f64]([0.75, -8.0, 1e-200])!
mantissa, exponent := values.frexp()!
reconstructed := vtl.ldexp(mantissa, exponent)!
println(mantissa.to_array())
println(exponent.to_array())
println(reconstructed.to_array())
```

The mantissa keeps the input f32/f64 type. Exponents use `Tensor[int]`.
`modf` and `frexp` reject non-floating tensors; `ldexp` rejects non-floating
mantissas. All three functions preserve logical tensor order, including views.

See the [math decomposition example](../examples/math_decomposition/README.md)
and the [NumPy parity tracker](./NUMPY_PARITY.md#math-and-ufuncs).