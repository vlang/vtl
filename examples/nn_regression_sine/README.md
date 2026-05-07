# nn_regression_sine

Trains a small MLP to approximate the `sin(x)` function over `[−π, π]`.

## What it demonstrates

- Building a regression model with VTL's `Sequential` API
- Using MSE loss for continuous-output regression
- Full-batch SGD training with the `sgd` optimizer
- Spot-checking predictions after training

## Architecture

```
Input(1) → Linear(16) → ReLU → Linear(1) → MSE Loss
```

## How to run

```sh
v run main.v
```

## Expected output

```
Training MLP to approximate sin(x)...
Epoch | Loss
------|----------
    0 | 0.251973
   10 | 0.205678
   ...
   59 | 0.141234

Spot-check after training:
  x       | sin(x)   | predicted
----------|----------|----------
  -1.5708 |  -1.0000 |  -0.9312
   0.0000 |   0.0000 |   0.0183
   1.5708 |   1.0000 |   0.9287
   3.1416 |   0.0000 |   0.0421
```

Loss decreases from ~0.25 to ~0.14 in 60 epochs.  
Predictions at key points closely match the true `sin(x)` values.

## Key parameters

| Parameter       | Value  | Notes                            |
|-----------------|--------|----------------------------------|
| `n_samples`     | 100    | Evenly spaced over [−π, π]       |
| `epochs`        | 60     | Full-batch gradient descent      |
| `learning_rate` | 0.001  | SGD step size                    |
| Hidden size     | 16     | Units in the single hidden layer |

## Related examples

- [`nn_simple_two_layer`](../nn_simple_two_layer/) — random-target regression, two hidden layers
- [`nn_multiclass_iris`](../nn_multiclass_iris/) — classification with softmax CE loss
- [`nn_autoencoder_simple`](../nn_autoencoder_simple/) — unsupervised reconstruction task
