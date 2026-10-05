# `vtl.nn.models`

`Sequential[T]` composes layers and a loss around an autograd context. Its
builder methods cover inputs, dense/convolutional layers, pooling, activations,
normalization, embeddings, recurrent and attention layers, and loss selection.
Models can also serialize and load weights.

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

The exact forward, loss, and serialization APIs are documented in
[`sequential.v`](sequential.v) and [`serialization.v`](serialization.v). Start
with the [neural-network tutorial](../../docs/TUTORIAL_NEURAL_NETWORKS.md).
