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

Complex tensors are an early part of VTL's complex-number support. Real-valued
casts, reductions, general complex mathematical functions, linear algebra,
random distributions, and `.npy`/`.npz` complex I/O do not yet support this
dtype. FFT APIs have their own complex output types and are documented in the
[FFT tutorial](./TUTORIAL_FFT.md).

Run the [complex tensor example](../examples/complex_tensors/README.md) from
`~/.vmodules`:

```sh
VJOBS=2 v run ./vtl/examples/complex_tensors/main.v
```
