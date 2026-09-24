# Transformer Attention

This example runs a small transformer-style self-attention block over a batch of
synthetic token embeddings. It demonstrates the two layers that make the block
position-aware:

1. **Positional encoding** adds a fixed sinusoidal vector for each sequence
   position. Without it, self-attention sees the tokens as an unordered set.
2. **Multi-head self-attention** projects the same input into queries, keys, and
   values. Each query compares with every key, and the resulting weights mix the
   values. The attention scores are scaled by `sqrt(head_dim)` before softmax.
   Multiple heads let the model learn different relationships between tokens.

The model receives tensors shaped `[batch, sequence length, embedding dimension]`
and returns the same shape. This demo uses a batch of two sequences, four tokens
per sequence, eight features per token, and two attention heads.

## How to run

From the repository root:

```sh
v run examples/nn_transformer_attention/main.v
```

Or from this directory:

```sh
v run main.v
```

The example prints the input and output shapes and the first output token. The
attention weights are initialized randomly, so the token values vary between
runs while the shapes remain stable.

## Architecture

```text
Input [2, 4, 8]
  -> sinusoidal positional encoding
  -> 2-head self-attention
Output [2, 4, 8]
```
