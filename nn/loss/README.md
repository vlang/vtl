# `vtl.nn.loss`

Differentiable losses consume a prediction `autograd.Variable[T]` and a target
`vtl.Tensor[T]`, returning a scalar-like variable suitable for `backprop()`.
Current losses include MSE, binary cross entropy, sigmoid/softmax cross
entropy, cross entropy, negative log likelihood, KL divergence, and Huber.

```v ignore
import vtl.nn.loss

objective := loss.mse_loss[f32]()
loss_value := objective.loss(prediction, target)!
```

Reduction and target conventions are function-specific; for example, cross
entropy variants expect different prediction representations. Read the
constructor comments and tests before switching a model's loss. The
[neural-network tutorial](../../docs/TUTORIAL_NEURAL_NETWORKS.md) demonstrates
training usage.
