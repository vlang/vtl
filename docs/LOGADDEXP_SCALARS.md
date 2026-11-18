# Scalar `logaddexp`

VTL provides scalar-operand forms of `logaddexp` for `f32` and `f64` tensors.
Both preserve the tensor's shape and dtype and use the same stable calculation
as the paired-tensor `logaddexp` operation:

```v
import vtl

values := vtl.from_1d([0.0, 1.0])!
tensor_right := vtl.logaddexp_scalar(values, 2.0)!
scalar_left := vtl.logaddexp_scalar_left(2.0, values)!
```

`logaddexp_scalar` computes `log(exp(x) + exp(scalar))` for each tensor value.
`logaddexp_scalar_left` puts the scalar first in the expression. Both operand
orders propagate NaNs and handle infinities using the established
`logaddexp` behavior.

The scalar must have the same type as the tensor (`f32` with `f32`, or `f64`
with `f64`). Mixed-dtype weak-scalar promotion is intentionally out of scope
for these APIs and remains owned by [#717](https://github.com/vlang/vtl/issues/717).

See the [runnable example](../examples/logaddexp_scalar/README.md) for both
operand orders.
