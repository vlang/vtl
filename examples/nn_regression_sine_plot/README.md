# nn_regression_sine_plot

Trains a small MLP to approximate `sin(x)` over [-π, π] and visualizes the
results using `vsl.plot`.

## What it demonstrates

- Generating synthetic training data (sampled sin curve)
- Defining a 2-hidden-layer MLP with `models.sequential_from_ctx`
- Full training loop with loss collection
- Plotting predicted vs true curves
- Plotting loss convergence over epochs

## Architecture

```
input(1) → linear(32) → relu → linear(32) → relu → linear(1) → mse_loss
```

## Run

```sh
v run .
```

Two interactive plots open in the browser:
1. True sin(x) vs MLP prediction
2. Training loss over 200 epochs
