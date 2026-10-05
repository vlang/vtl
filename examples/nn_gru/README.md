# Sequential GRU

This example builds a single-layer GRU through `vtl.nn.models.Sequential`,
passes a three-step sequence through it, and runs autograd backward.

Input and output tensors use `[sequence, batch, features]` layout. The layer
starts each forward pass with a zero hidden state; its output includes the
hidden state at every timestep.

Run from the V module root:

```sh
v run vtl/examples/nn_gru/main.v
```

The example uses only the CPU backend. See the
[neural-network tutorial](../../docs/TUTORIAL_NEURAL_NETWORKS.md#recurrent-gru-layer)
for the API and the [models reference](../../nn/models/README.md) for
serialization behavior.
