# `vtl.ml`

Machine-learning utilities currently include metrics under `vtl.ml.metrics`;
see the [metrics guide](metrics/README.md) for examples.
Training building blocks such as layers, losses, optimizers, and datasets live
in their own modules; this package is not a complete estimator or preprocessing
framework.

```v
import vtl
import vtl.ml.metrics

predicted := vtl.from_1d[f64]([1.0, 3.0])!
expected := vtl.from_1d[f64]([2.0, 2.0])!
mae := metrics.mean_absolute_error[f64](predicted, expected)!
```

Metrics include accuracy, squared/absolute/relative errors and their means.
Inputs must have compatible shapes; consult each function for its precise
semantics. Dataset loading and end-to-end examples are indexed in
[`examples/README.md`](../examples/README.md).
