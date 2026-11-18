# Scalar `logaddexp`

Demonstrates `logaddexp` with a scalar on either side of an `f64` tensor.

## Run

From the VTL repository root:

```sh
v run ./examples/logaddexp_scalar/main.v
```

These APIs require the scalar and tensor to have the same `f32` or `f64`
type. Mixed-dtype weak-scalar promotion is out of scope and remains owned by
[issue #717](https://github.com/vlang/vtl/issues/717).

See [the scalar `logaddexp` guide](../../docs/LOGADDEXP_SCALARS.md) for details.
