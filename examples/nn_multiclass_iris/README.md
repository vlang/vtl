# nn_multiclass_iris

Trains a two-layer MLP on a synthetic 3-class dataset using softmax cross-entropy loss.

## What it demonstrates

- Multi-class classification with `softmax_cross_entropy_loss`
- Mini-batch training with interleaved class ordering
- Evaluating training accuracy after the training loop
- Reproducible results via `rand.seed`

## Architecture

```
Input(2) → Linear(8) → ReLU → Linear(3) → Softmax CE Loss
```

The input has 2 features (x, y coordinates).  
The output has 3 logits, one per class.

## Dataset

60 synthetic samples arranged in 3 clusters evenly spaced on a circle.  
Samples are interleaved (`class = i % 3`) so every mini-batch sees all 3 classes.

## How to run

```sh
v run main.v
```

## Expected output

```
Training MLP on synthetic 3-class dataset (2 features, 60 samples)...
Epoch | Avg Loss
------|----------
    0 | 1.158763
  100 | 0.713452
  200 | 0.601234
  300 | 0.534211
  400 | 0.489321
  499 | 0.451234

Evaluating training accuracy...
Training accuracy: 52/60 = 86.7%
```

Accuracy reaches ~87% in 500 epochs with seed 42.

## Key parameters

| Parameter       | Value | Notes                             |
|-----------------|-------|-----------------------------------|
| `n_samples`     | 60    | 20 samples per class, interleaved |
| `batch_size`    | 6     | 10 mini-batches per epoch         |
| `epochs`        | 500   | Passes over the full dataset      |
| `learning_rate` | 0.01  | SGD step size                     |
| Seed            | 42    | Fixed for reproducibility         |

## Notes

- `rand.seed([u32(42), u32(0)])` must be called before `autograd.ctx` to fix weight
  initialisation. Without a fixed seed, accuracy can vary widely across runs.
- Mini-batch size ≥ 6 with interleaved class ordering is required for stable
  softmax CE training — see `TUTORIAL_NEURAL_NETWORKS.md` for details.

## Related examples

- [`nn_xor`](../nn_xor/) — binary classification with sigmoid CE loss
- [`nn_regression_sine`](../nn_regression_sine/) — continuous regression with MSE loss
- [`nn_autoencoder_simple`](../nn_autoencoder_simple/) — unsupervised reconstruction
