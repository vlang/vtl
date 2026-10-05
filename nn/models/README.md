# `vtl.nn.models`

`Sequential[T]` composes layers and a loss around an autograd context. Its
builder methods cover inputs, dense/convolutional layers, pooling, activations,
normalization, embeddings, recurrent and attention layers, and loss selection.
The architecture is an ordered list: each builder adds a layer whose output
feeds the next layer. The context is supplied when constructing the model.

`gru(input_size, hidden_size)` adds a single-layer CPU GRU. It uses a zero
initial hidden state and expects `[sequence, batch, features]` input. Its output
keeps the sequence and batch dimensions and replaces the feature dimension with
`hidden_size`.

```v ignore
import vtl.autograd
import vtl.nn.models

ctx := autograd.ctx[f32]()
mut model := models.sequential_from_ctx[f32](ctx)
model.input([4])
model.linear(8)
model.relu()
model.linear(2)
model.mse_loss()
```

Use `forward` for inference and the model's loss/backpropagation methods for
training; consult [`sequential.v`](sequential.v) for exact method signatures.

`save(path)` writes model weights. `save_checkpoint(path, epoch, loss)` also
stores epoch/loss metadata; `load_weights(path)` loads into an already-built
compatible model and `Sequential.load_checkpoint[T](path)` reads checkpoint
metadata. The JSON format has version `1.0`, ordered layer definitions,
per-layer weight maps, optimizer metadata, and checkpoint metadata; tensor
weights are base64 encoded. Loading rejects a version mismatch. The current
save API does not promise restoration of optimizer moment buffers, so treat it
as a model-weight checkpoint rather than a full training-state snapshot.

Start with the [neural-network tutorial](../../docs/TUTORIAL_NEURAL_NETWORKS.md).
