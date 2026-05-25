# vtl.ml.metrics

Utilities for evaluating classification and regression outputs.

## Available metrics

- `accuracy_score(y_pred, y_true)`
- `squared_error(y, y_true)`
- `mean_squared_error(y, y_true)`
- `absolute_error(y, y_true)`
- `mean_absolute_error(y, y_true)`
- `relative_error(y, y_true)`
- `mean_relative_error(y, y_true)`

## Examples

Runnable examples are available in [`examples/`](./examples/):

- `classification_accuracy`
- `regression_errors`

Run pattern:

```sh
v run ml/metrics/examples/<example_name>/main.v
```
