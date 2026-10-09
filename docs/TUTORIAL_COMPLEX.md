# Complex tensors

VTL stores `math.complex.Complex` values as the `complex128` tensor dtype.
Creation, indexing, views, broadcasting, and elementwise addition, subtraction,
multiplication, and division use the same tensor APIs as real-valued data.

```v
import math.complex as cmplx
import vtl

values := vtl.from_1d([
	cmplx.complex(1.0, 2.0),
	cmplx.complex(3.0, -1.0),
])!

println(values.dtype()) // complex128
println(values.add(values)!)
println(values.multiply(values)!)
```

Use `promote_types` to query the result dtype when combining dtype metadata:

```v
import vtl

assert vtl.promote_types(.complex128, .float64)! == .complex128
assert vtl.promote_types(.int32, .complex128)! == .complex128
```

`vtl.la.matmul` supports complex128 vectors, matrices, and broadcast batches.
As in NumPy, vector `matmul` does not conjugate either operand:

```v
import math.complex as cmplx
import vtl
import vtl.la

left := vtl.from_1d([cmplx.complex(1.0, 1.0), cmplx.complex(2.0, 0.0)])!
right := vtl.from_1d([cmplx.complex(0.0, 1.0), cmplx.complex(1.0, 0.0)])!
product := la.matmul(left, right)!
assert product.rank() == 0
assert product.get_nth(0) == cmplx.complex(1.0, 1.0)
```

`vtl.la.solve_complex` solves complex128 systems with partial pivoting. The
right-hand side may be a vector or a matrix, and leading batch dimensions
broadcast like NumPy. `vtl.la.inv_complex` applies the same solve to complex
identity matrices and preserves leading batches. Singular systems return an
error. See the [complex linear solve example](../examples/complex_linear_solve/README.md).

```v
import math
import math.complex as cmplx
import vtl
import vtl.la

matrix := vtl.from_array[cmplx.Complex]([
	cmplx.Complex{ re: 1, im: 1 }, cmplx.Complex{ re: 2 },
	cmplx.Complex{ re: 0, im: 1 }, cmplx.Complex{ re: 3 },
], [2, 2])!
inverse := la.inv_complex(matrix)!
identity := la.matmul(matrix, inverse)!
assert math.abs(identity.get([0, 0]).re - 1.0) < 1e-12
```

The NumPy-style `real`, `imag`, `conj`, `absolute`, and `abs` operations are
available.
These functions return new tensors with the input shape. Component and magnitude
outputs use float64, while the complex-valued functions preserve the complex
dtype:

```v
import math.complex as cmplx
import vtl

values := vtl.from_1d([cmplx.complex(3.0, 4.0), cmplx.complex(-5.0, 12.0)])!
assert vtl.real(values)!.to_array() == [3.0, -5.0]
assert vtl.imag(values)!.to_array() == [4.0, 12.0]
assert vtl.conj(values)!.to_array()[0].im == -4.0
assert vtl.absolute(values)!.to_array() == [5.0, 13.0]
assert vtl.abs(values)!.to_array() == [5.0, 13.0]

origin := vtl.from_1d([cmplx.complex(0.0, 0.0), cmplx.complex(-4.0, 0.0)])!
assert vtl.exp(origin)!.get_nth(0).re == 1.0
assert vtl.sqrt(origin)!.get_nth(1).im == 2.0
assert vtl.sin(origin)!.get_nth(0).re == 0.0
assert vtl.cos(origin)!.get_nth(0).re == 1.0
```

`complex_angle` returns each value's phase in radians, or degrees when its
second argument is `true`. `complex_is_nan`, `complex_is_inf`, and
`complex_is_finite` inspect either or both components using NumPy's complex
predicate behavior:

```v
import math
import math.complex as cmplx
import vtl

values := vtl.from_1d([
	cmplx.complex(0.0, 1.0),
	cmplx.complex(math.nan(), 0.0),
	cmplx.complex(math.inf(1), 0.0),
])!
assert vtl.complex_angle(values, true)!.to_array()[0] == 90.0
assert vtl.complex_is_nan(values)!.to_array() == [false, true, false]
assert vtl.complex_is_inf(values)!.to_array() == [false, false, true]
assert vtl.complex_is_finite(values)!.to_array() == [true, false, false]
```

`log` uses the principal natural logarithm, and `sqrt` uses the principal
square root branch. `arcsin`, `arccos`, `arctan`, `arcsinh`, `arccosh`, and
`arctanh` use V's standard complex library for their principal inverse branches.
The direct `tan`, `sinh`, `cosh`, and `tanh` functions are also available.

Global sum, product, and mean use complex arithmetic and return scalar complex
values. Single- and multi-axis sum/product preserve complex dtype and shape.
Empty means return `NaN + NaN·i`, matching NumPy's invalid empty mean:

```v
import math.complex as cmplx
import vtl
import vtl.stats

values := vtl.from_1d([cmplx.complex(1.0, 2.0), cmplx.complex(3.0, 4.0)])!
assert stats.sum(values) == cmplx.Complex{ re: 4.0, im: 6.0 }
assert stats.prod(values) == cmplx.Complex{ re: -5.0, im: 10.0 }
assert stats.mean(values) == cmplx.Complex{ re: 2.0, im: 3.0 }
```

Axis-wise complex means preserve complex values. Variance and standard deviation
return real tensors using `E(|z - mean(z)|²)`, and accept one or several axes,
negative axes, `keepdims`, and `ddof`:

```v
import math.complex as cmplx
import vtl
import vtl.stats

values := vtl.from_array[cmplx.Complex]([
	cmplx.complex(1.0, 2.0),
	cmplx.complex(3.0, 4.0),
	cmplx.complex(5.0, 6.0),
	cmplx.complex(7.0, 8.0),
], [2, 2])!
row_means := stats.complex_mean_along_axis(values, 1, false)!
row_variances := stats.complex_variance_along_axis(values, 1, 0, false)!
assert row_means.shape == [2]
assert row_variances.to_array() == [2.0, 2.0]
```

Complex tensors are an early part of VTL's complex-number support. Real-valued
casts, complex linear algebra, and random distributions do not yet support this
dtype. `.npy` and `.npz` round trips support NumPy `complex128` arrays. FFT APIs
have their own complex output types and are documented in the
[FFT tutorial](./TUTORIAL_FFT.md).

Run the [complex tensor example](../examples/complex_tensors/README.md) from
`~/.vmodules`:

```sh
VJOBS=2 v run ./vtl/examples/complex_tensors/main.v
```
