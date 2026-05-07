# nn_xor

Trains an MLP to learn the XOR function using sigmoid cross-entropy loss.

## What it demonstrates

- Binary classification with `sigmoid_cross_entropy_loss`
- Mini-batch training over multiple epochs
- Generating synthetic boolean training data with `vtl.random`
- The classic XOR problem as a non-linearly-separable dataset

## Architecture

```
Input(2) → Linear(3) → ReLU → Linear(1) → Sigmoid CE Loss
```

## Dataset

32 000 random boolean pairs `(a, b)` (batch_size=32, batches=1000).  
Target: `a XOR b` — outputs 1 when exactly one input is 1.

## How to run

```sh
v run main.v
```

## Expected output

```
Epoch: 0
Epoch: 0, Batch id: 0, Loss: [[0.654321]]
...
Epoch: 5
Epoch: 5, Batch id: 99, Loss: [[0.198765]]
```

Loss typically decreases from ~0.65 toward ~0.20 over 6 epochs × 100 batches.

## Key parameters

| Parameter       | Value | Notes                            |
|-----------------|-------|----------------------------------|
| `batch_size`    | 32    | Samples per mini-batch           |
| `batches`       | 100   | Mini-batches per epoch           |
| `epochs`        | 6     | Full passes over the dataset     |
| `learning_rate` | 0.01  | SGD step size                    |

## Notes

XOR is not linearly separable, so a hidden layer with non-linear activation
(ReLU here) is required. A linear model would fail to learn XOR.

## Related examples

- [`nn_multiclass_iris`](../nn_multiclass_iris/) — multi-class softmax CE classification
- [`nn_regression_sine`](../nn_regression_sine/) — continuous regression with MSE loss
- [`autograd_backprop`](../autograd_backprop/) — raw autograd without a Sequential model
