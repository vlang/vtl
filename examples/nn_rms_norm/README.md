# RMSNorm Layer

This example applies RMSNorm to the last dimension of a batch of two synthetic
four-feature vectors. The layer learns one scale per feature and omits the
mean-centering and bias used by LayerNorm.

Run it from `~/.vmodules`:

```sh
v run ./vtl/examples/nn_rms_norm/main.v
```

The input and output shapes are both `[2, 4]`.
