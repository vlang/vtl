# GroupNorm Layer

This example builds a channel-first GroupNorm layer in `Sequential` for a batch
of four-channel, 8 × 8 feature maps. Two groups normalize two channels each;
the default affine mode learns a scale and bias for every channel.

## Run

From `~/.vmodules`:

```sh
v run ./vtl/examples/nn_group_norm/main.v
```

Expected output shapes are `[2, 4, 8, 8]` for both input and output. The layer
contains eight trainable scalar parameters: four scales and four biases.
