# nn_simple_two_layer

Trains a two-layer MLP to fit random targets using MSE loss.

## What it demonstrates

- Building a `Sequential` model with `vtl.nn.models`
- Registering model parameters with `optimizers.sgd`
- Running a basic training loop: forward → loss → backprop → update
- Collecting loss values across epochs

## Architecture

```
Input(100) → Linear(32) → ReLU → Linear(10) → MSE Loss
```

## Dataset

Random tensors: `x` of shape `[64, 100]` and `y` of shape `[64, 10]`.  
There is no true structure — the model is simply fitting random noise.

## How to run

```sh
v run main.v
```

## Expected output

```
Epoch: 0, Loss: [[0.321456]]
Epoch: 1, Loss: [[0.318234]]
...
Epoch: 19, Loss: [[0.287654]]
```

Loss decreases slowly because the target is random (no learnable structure).

## Key parameters

| Parameter       | Value | Notes                           |
|-----------------|-------|---------------------------------|
| `batch_size`    | 64    | Number of training samples      |
| `input_dim`     | 100   | Input feature dimensionality    |
| `hidden_dim`    | 32    | Units in the hidden layer       |
| `output_dim`    | 10    | Output dimensionality           |
| `epochs`        | 20    | Training iterations             |
| `learning_rate` | 1e-3  | SGD step size                   |

## Related examples

- [`nn_regression_sine`](../nn_regression_sine/) — structured regression (sin function)
- [`nn_xor`](../nn_xor/) — binary classification with mini-batching
- [`autograd_backprop`](../autograd_backprop/) — raw autograd API
