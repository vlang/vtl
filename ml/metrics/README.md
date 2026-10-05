# `vtl.ml.metrics`

Metrics compare tensor predictions with reference values. The module currently
provides elementwise squared, absolute, and relative error tensors; their mean
forms; and `accuracy_score` for elementwise-equal labels.

```v
import vtl
import vtl.ml.metrics

predicted := vtl.from_1d[f64]([1.0, 3.0, 1.0, 2.0])!
expected := vtl.from_1d[f64]([1.0, 2.0, 1.0, 0.0])!
accuracy := metrics.accuracy_score(predicted, expected)!
mae := metrics.mean_absolute_error[f64](predicted, expected)!
```

`accuracy_score` compares corresponding tensor elements and returns the fraction
that match. It expects class IDs or labels already represented in comparable
tensors; it does not apply `argmax` to logits. Error functions compare
corresponding numeric values. See [`metrics_test.v`](metrics_test.v) for
executable examples and edge cases.
