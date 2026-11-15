# Elementwise Math and NumPy Ufuncs

VTL provides elementwise math operations on tensors, including decomposition of
floating-point values into integer/fractional parts or binary mantissas and
exponents.

## Square

`Tensor.square()` returns an elementwise square in the input dtype and shape.
It supports numeric, boolean, and complex tensors:

```v
import vtl

values := vtl.from_1d[f64]([1.5, -2.0])!
squared := values.square()!
println(squared.to_array()) // [2.25, 4.0]
```

## Promoted square and cube roots

`vtl.sqrt_promoted[R, T](values)` and `vtl.cbrt_promoted[R, T](values)` return
floating-point square and cube roots. They return f64 for f64 or
32-bit-and-wider integer inputs, and f32 otherwise. NumPy's float16 results for
booleans and 8-bit integers are widened because VTL does not support float16.
The output type is explicit and validated:

```v
import vtl

values := vtl.from_1d[i32]([1, 2, 9])!
roots := vtl.sqrt_promoted[f64, i32](values)!
println(roots.to_array()) // [1.0, 1.414..., 3.0]

signed_values := vtl.from_1d[i16]([-8, 0, 27])!
cube_roots := vtl.cbrt_promoted[f32, i16](signed_values)!
println(cube_roots.to_array()) // [-2.0, 0.0, 3.0]
```

Complex and string inputs are unsupported. The same-dtype `Tensor.sqrt()` and
`Tensor.cbrt()` methods remain available for compatibility; use the promoted
functions when NumPy-style integer output is needed.

## Promoted exponential and logarithm

`vtl.exp_promoted[R, T](values)` and `vtl.log_promoted[R, T](values)`, along
with `exp2_promoted`, `expm1_promoted`, `log2_promoted`, `log10_promoted`, and
`log1p_promoted`, share the promoted real-output dtype rules: f64 for f64 or
32-bit-and-wider integer inputs, and f32 otherwise. They preserve IEEE domain
results, such as `-inf` for `log(0)` and `NaN` for logarithms of negative
values. `expm1` and `log1p` retain precision near zero.

```v
import vtl

values := vtl.from_1d[i16]([0, 1])!
exponents := vtl.exp_promoted[f32, i16](values)!
logs := vtl.log_promoted[f32, i16](values)!
println(exponents.to_array())
println(logs.to_array()) // [-inf, 0.0]
```

## Promoted rounding

`rint_promoted` rounds ties to even and uses the same floating output dtype
rules as `sqrt_promoted`. `floor_promoted`, `ceil_promoted`, and
`trunc_promoted` preserve their input dtype, including booleans and integers;
integer inputs are copied without conversion through floating point.

```v
import vtl

values := vtl.from_1d[f64]([-2.5, -1.5, 1.5, 2.5])!
rounded := vtl.rint_promoted[f64, f64](values)!
floored := vtl.floor_promoted[f64, f64](values)!
println(rounded.to_array()) // [-2.0, -2.0, 2.0, 2.0]
println(floored.to_array()) // [-3.0, -2.0, 1.0, 2.0]
```

## Promoted trigonometric functions

`vtl.sin_promoted[R, T]`, `vtl.cos_promoted[R, T]`, and
`vtl.tan_promoted[R, T]` apply the trigonometric functions elementwise and use
the same promoted output dtype rules. Angles are in radians.
`degrees_promoted` and `radians_promoted` convert angle units with those same
output dtype rules.

```v
import math
import vtl

angles := vtl.from_1d[f32]([0.0, f32(math.pi / 4.0)])!
sines := vtl.sin_promoted[f32, f32](angles)!
cosines := vtl.cos_promoted[f32, f32](angles)!
tangents := vtl.tan_promoted[f32, f32](angles)!
println(sines.to_array())
println(cosines.to_array())
println(tangents.to_array())
```

The promoted inverse trigonometric APIs are `asin_promoted`, `acos_promoted`,
and `atan_promoted`; the hyperbolic counterparts are `sinh_promoted`,
`cosh_promoted`, and `tanh_promoted`. Their inverse hyperbolic counterparts
are `asinh_promoted`, `acosh_promoted`, and `atanh_promoted`. All preserve the
floating-point domain behavior of the underlying operations; the trigonometric
functions take angles in radians.

## Promoted arctangent with two arguments

`vtl.atan2_promoted[R, T, U](y, x)` broadcasts real numeric or boolean tensors
and returns the floating-point result. It returns f64 if either input is f64 or a
32-bit-or-wider integer; otherwise it returns f32. NumPy's float16 results for
boolean and 8-bit integer inputs are widened because VTL does not support
float16. Since V requires statically selected generic types, callers specify
the output and both input types; mismatches return an error:

```v
import vtl

y := vtl.from_1d[i32]([1, 0])!
x := vtl.from_1d[f64]([1.0, -1.0])!
angles := vtl.atan2_promoted[f64, i32, f64](y, x)!
println(angles.to_array()) // [0.785398..., 3.141592...]
```

Complex and string inputs are unsupported. The same-dtype
`Tensor.atan2()` method remains available for compatibility; use the promoted
function when NumPy-style integer or mixed-dtype output is needed.

## Promoted binary numeric functions

`copysign_promoted`, `heaviside_promoted`, `hypot_promoted`,
`logaddexp_promoted`, `logaddexp2_promoted`, and `nextafter_promoted` broadcast
real numeric or boolean operands and use the same floating output dtype rules
as `atan2_promoted`. The f32 `nextafter_promoted` path steps at f32 precision.
The log-add-exp variants use stable calculations for large-magnitude inputs:

```v
import math
import vtl

a := vtl.from_1d[i16]([1000, -1000])!
b := vtl.from_1d[f32]([1000.0, -1000.0])!
values := vtl.logaddexp_promoted[f32, i16, f32](a, b)!
println(values.to_array())

x := vtl.from_1d[i16]([0, 1])!
toward := vtl.from_1d[f32]([1.0, -math.inf(1)])!
steps := vtl.nextafter_promoted[f32, i16, f32](x, toward)!
println(steps.to_array())

x1 := vtl.from_1d[f32]([-1.0, 0.0, 1.0])!
x2 := vtl.from_1d[i8]([0, 2, 0])!
heaviside_result := vtl.heaviside_promoted[f32, f32, i8](x1, x2)!
println(heaviside_result.to_array())
```

## Floor division

`Tensor.floor_divide()` broadcasts same-dtype real integer or floating-point
tensors and rounds the quotient toward negative infinity, unlike integer
division that truncates toward zero. `floor_divide_scalar()` applies the same
operation to a scalar while preserving the tensor's dtype:

```v
import vtl

numerators := vtl.from_1d([-3, 3])!
divisors := vtl.from_1d([2, -2])!
quotients := numerators.floor_divide(divisors)!
println(quotients.to_array()) // [-2, -2]
```

Integer zero divisors produce zero, matching NumPy's result values (VTL does
not emit NumPy's runtime warning). Floating-point zero divisors follow IEEE
behavior. Mixed-dtype operands and automatic result-type selection remain part
of [#717](https://github.com/vlang/vtl/issues/717). Boolean and complex tensors
are unsupported. See the [parity tracker](./NUMPY_PARITY.md#math-and-ufuncs).

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

## Stable log-domain operations and sign bits

`vtl.logaddexp(a, b)` computes `log(exp(a) + exp(b))` without overflowing
intermediate exponentials. `vtl.logaddexp2` computes the base-2 equivalent,
`log2(2^a + 2^b)`. Both accept same-type f32/f64 tensors, broadcast operands,
preserve dtype, and handle infinities and NaNs.

`Tensor.signbit()` returns a boolean tensor identifying negative floating-point
values, including negative zero and negative infinity. Signed integers are true
when negative; unsigned integers and booleans return false for every element.
`vtl.copysign(magnitude, sign)` combines the magnitude of its first f32/f64
tensor with the sign bit of its second, with standard broadcasting.
`vtl.copysign_promoted` additionally accepts mixed real numeric and boolean
inputs and applies the same NumPy-style float output promotion as
`atan2_promoted`, including negative-zero and NaN sign bits.

```v
import vtl

logits := vtl.from_1d[f64]([-1000.0, -1001.0])!
normalizer := vtl.from_1d[f64]([-1000.0])!
log_sum := vtl.logaddexp(logits, normalizer)!
signed := vtl.copysign(logits.abs(), logits)!
println(log_sum.to_array())
println(signed.to_array())
```

See the [stable log-domain example](../examples/math_log_domain/README.md) and
the [NumPy parity tracker](./NUMPY_PARITY.md#math-and-ufuncs).
