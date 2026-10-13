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
metadata. For resumable Adam training, build the optimizer with the model's
parameters and use `save_training_checkpoint(path, epoch, loss, optimizer)`;
restore into a compatible model and a built Adam optimizer with
`load_training_checkpoint(path, mut optimizer)`. This stores Adam's scalar
configuration and first/second moments alongside model weights and metadata.
For SGD with momentum, use `save_sgd_training_checkpoint` and
`load_sgd_training_checkpoint`; these store the scalar configuration, momentum
buffers, and first-step initialization state. For AdamW, use
`save_adamw_training_checkpoint` and `load_adamw_training_checkpoint` to store
its scalar configuration and first/second moments. For RMSProp, use
`save_rmsprop_training_checkpoint` and `load_rmsprop_training_checkpoint`;
these store its scalar configuration and squared-gradient buffers. Other
optimizer types, device remapping, and cross-version migration are not
supported by this API yet. The JSON format has version `1.0`, ordered layer
definitions, per-layer weight maps, optimizer metadata, and checkpoint
metadata; tensor weights are base64 encoded. Loading rejects a version
mismatch.

```v ignore
import vtl.nn.models
import vtl.nn.optimizers

mut model := models.sequential[f64]()
model.input([2])
model.linear(1)
mut optimizer := optimizers.adam_optimizer[f64](learning_rate: 0.001)
optimizer.build_params(model.info.layers)
model.save_training_checkpoint('training.json', 12, 0.25, optimizer)!

mut resumed := models.sequential[f64]()
resumed.input([2])
resumed.linear(1)
mut resumed_optimizer := optimizers.adam_optimizer[f64](learning_rate: 0.001)
resumed_optimizer.build_params(resumed.info.layers)
epoch, loss := resumed.load_training_checkpoint('training.json', mut resumed_optimizer)!
```

Start with the [neural-network tutorial](../../docs/TUTORIAL_NEURAL_NETWORKS.md).
