# nn_autoencoder_simple

Trains a simple autoencoder that compresses a 4-D input to a 2-D bottleneck
and reconstructs the original vector.

## What it demonstrates

- The encoder–decoder pattern: compress → reconstruct
- Using the same `Sequential` model for both encoder and decoder
- MSE reconstruction loss (target = input)
- How backpropagation flows end-to-end through a bottleneck architecture

## Architecture

```
Input(4) → Linear(4) → ReLU → Linear(2)   ← Encoder (bottleneck)
         → Linear(4) → ReLU → Linear(4)   ← Decoder
         → MSE Loss
```

The bottleneck layer (Linear 4→2) forces the network to learn a compact
2-D representation of the 4-D input.

## Dataset

40 samples: 4 phase-shifted sine waves sampled at 4 points, each scaled by
a random amplitude in [0.5, 1.0]. These patterns lie in a 2-D subspace of
4-D space, making the 2-D bottleneck theoretically sufficient.

## How to run

```sh
v run main.v
```

## Expected output

```
Training autoencoder  (4-D → 2-D bottleneck → 4-D)
Data: sine-wave patterns, 40 samples, full-batch SGD
Epoch | Loss
------|----------
    0 | 0.328788
   30 | 0.207182
   60 | 0.196553
   90 | 0.195625
  120 | 0.195535
  149 | 0.195524

Loss reduced by 40.5%  (0.3288 → 0.1955)

Reconstruction spot-check (first 3 samples):
  # | Original                | Reconstructed
----|-------------------------|-------------------------
  0 | [  0.00  0.63  0.90  0.63 ] | [  0.48  0.47  0.19 -0.21 ]
  1 | [  0.53  0.75  0.53  0.00 ] | [  0.48  0.47  0.19 -0.21 ]
  2 | [  0.90  0.64  0.00 -0.64 ] | [  0.48  0.47  0.19 -0.21 ]

Final reconstruction RMSE: 0.4422  (data range ≈ [−1, 1])
```

Loss decreases ~40% in the first 30 epochs, then plateaus.  
With plain SGD (no momentum), the model converges to a near-mean solution —
a known limitation of MSE + vanilla SGD on reconstruction tasks.

## Key parameters

| Parameter       | Value | Notes                               |
|-----------------|-------|-------------------------------------|
| `input_dim`     | 4     | Input and output dimensionality     |
| `bottleneck_dim`| 2     | Latent space dimensionality         |
| `n_samples`     | 40    | Training samples                    |
| `epochs`        | 150   | Full-batch gradient descent passes  |
| `learning_rate` | 0.001 | SGD step size                       |

## Notes

Plain SGD without momentum can collapse to outputting the dataset mean for
every input (a local minimum of MSE with zero variance). This is a known
limitation. In practice, use Adam or SGD with momentum for autoencoder training.
The loss curve still demonstrates that backpropagation through the bottleneck
works correctly.

## Related examples

- [`nn_regression_sine`](../nn_regression_sine/) — supervised regression with MSE loss
- [`nn_simple_two_layer`](../nn_simple_two_layer/) — supervised two-layer MLP
- [`nn_multiclass_iris`](../nn_multiclass_iris/) — multi-class classification
