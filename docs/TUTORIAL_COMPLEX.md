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

`log` uses the principal natural logarithm, and `sqrt` uses the principal
square root branch. `arcsin`, `arccos`, `arctan`, `arcsinh`, `arccosh`, and
`arctanh` use V's standard complex library for their principal inverse branches.
The direct `tan`, `sinh`, `cosh`, and `tanh` functions are also available.

Global sum, product, and mean use complex arithmetic and return scalar complex
values. Empty means return `NaN + NaN·i`, matching NumPy's invalid empty mean:

```v
import math.complex as cmplx
import vtl
import vtl.stats

values := vtl.from_1d([cmplx.complex(1.0, 2.0), cmplx.complex(3.0, 4.0)])!
assert stats.sum(values) == cmplx.Complex{ re: 4.0, im: 6.0 }
assert stats.prod(values) == cmplx.Complex{ re: -5.0, im: 10.0 }
assert stats.mean(values) == cmplx.Complex{ re: 2.0, im: 3.0 }
```

Complex tensors are an early part of VTL's complex-number support. Real-valued
casts, axis-wise reductions, other statistics, linear algebra, random
distributions, and `.npy`/`.npz` complex I/O do not yet support this dtype.
Global sum, product, and mean return complex scalar values. FFT APIs have their
own complex output types and are documented in the [FFT tutorial](./TUTORIAL_FFT.md).

Run the [complex tensor example](../examples/complex_tensors/README.md) from
`~/.vmodules`:

```sh
VJOBS=2 v run ./vtl/examples/complex_tensors/main.v
```
