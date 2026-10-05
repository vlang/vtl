# Self-attention with positional encoding

This small CPU example composes VTL's sinusoidal positional encoding and
multi-head self-attention layers. The input is a batch of four token feature
vectors, each with an embedding dimension of eight. Positional encoding adds
order information before attention compares each token with the sequence.

The attention layer forms query, key, and value projections, computes scaled
dot-product attention over the sequence, combines the heads, and projects the
result back to the embedding dimension. The program prints input and output
shapes plus one output value so the forward path can be checked quickly.

Run it from `~/.vmodules`:

```sh
v run vtl/examples/nn_transformer_attention/main.v
```

Expected output shape: `[1, 4, 8]`.
