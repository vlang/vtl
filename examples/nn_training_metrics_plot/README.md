# nn_training_metrics_plot

Trains a classifier on the XOR problem and visualizes training metrics
(loss and accuracy) over epochs using `vsl.plot`.

## What it demonstrates

- Defining a classification model with VTL's Sequential API
- Computing accuracy during training (threshold at 0.5)
- Collecting per-epoch metrics into arrays
- Plotting loss and accuracy curves on the same chart
- Final evaluation with individual predictions

## Architecture

```
input(2) → linear(8) → relu → linear(8) → relu → linear(1) → mse_loss
```

## Run

```sh
v run .
```

An interactive plot opens showing loss decrease and accuracy increase over
300 epochs, followed by final predictions printed to the console.
