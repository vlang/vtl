# Getting Started with VTL

This guide walks you through VTL from zero to training a neural network.
By the end you will understand how tensors, autograd, and the neural network
module work together.

## Prerequisites

Install V and VSL following the [main README](../README.md#installation).

## 1. Create a tensor

```v
import vtl

// From a V array
t := vtl.from_2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])!
println(t.shape) // [2, 3]
println(t.size) // 6
println(t.rank()) // 2
```

Tensors are the fundamental data structure — n-dimensional arrays that support
element-wise arithmetic, slicing, reshaping, and broadcasting.

See: [First Steps](./TUTORIAL_FIRST_STEPS.md) for full tensor creation API.

## 2. Do math on tensors

```v
import vtl

a := vtl.from_1d([1.0, 2.0, 3.0])!
b := vtl.from_1d([4.0, 5.0, 6.0])!

c := a.add(b)! // [5, 7, 9]
d := a.multiply(b)! // [4, 10, 18]
println(c)
println(d)
```

VTL supports element-wise ops, broadcasting, reductions (sum, mean, argmax),
and linear algebra (matmul, solve, decompositions).

See: [Broadcasting](./TUTORIAL_BROADCASTING.md), [Linear Algebra](./TUTORIAL_LINEAR_ALGEBRA.md)

## 3. Automatic differentiation

The autograd engine tracks operations on `Variable` wrappers and computes
gradients via backpropagation:

```v
import vtl
import vtl.autograd

ctx := autograd.ctx[f64]()

x := ctx.variable(vtl.from_1d([3.0])!)
y := ctx.variable(vtl.from_1d([2.0])!)

mut z := x.pow(y)! // z = 3^2 = 9
z.backprop()!

println(x.grad) // [6.0] — dz/dx = 2*x = 6
```

This is how neural networks learn — the loss is backpropagated through
the entire computation graph to update weights.

See: [Autograd Tutorial](./TUTORIAL_AUTOGRAD.md)

## 4. Train a neural network

VTL's `Sequential` model API lets you define, train, and evaluate networks
in a few lines:

```v ignore
import vtl
import vtl.autograd
import vtl.nn.models
import vtl.nn.optimizers

ctx := autograd.ctx[f64]()

// Define: input(2) → hidden(8) → relu → output(1)
mut model := models.sequential_from_ctx[f64](ctx)
model.input([2])
model.linear(8)
model.relu()
model.linear(1)
model.mse_loss()

// Optimizer
mut opt := optimizers.sgd[f64](learning_rate: 0.01)
opt.build_params(model.info.layers)

// Training loop
x_tensor := vtl.from_array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], [4, 2])!
y_tensor := vtl.from_array([0.0, 1.0, 1.0, 0.0], [4, 1])!
mut x := ctx.variable(x_tensor, requires_grad: true)

for epoch in 0 .. 300 {
    y_pred := model.forward(x)!
    mut loss := model.loss(y_pred, y_tensor)!
    loss.backprop()!
    opt.update()!
}
```

See: [Neural Networks](./TUTORIAL_NEURAL_NETWORKS.md), [Optimizers](./TUTORIAL_OPTIMIZERS.md)

## 5. Visualize results

VTL integrates with `vsl.plot` for interactive training visualization:

```v ignore
import vsl.plot

mut plt := plot.Plot.new()
plt.scatter(x: epoch_x, y: losses, mode: 'lines', name: 'Loss')
plt.layout(title: 'Training Convergence')
plt.show()!
```

See: [Visualizing Training](./TUTORIAL_NEURAL_NETWORKS.md#visualizing-training-with-vslplot)

## 6. GPU acceleration (optional)

Compile with `-d vulkan` or `-d vcl` to offload computation to GPU:

```sh
v -d vulkan run my_model.v
VTL_BACKEND=vulkan v -d vulkan run my_model.v  # force Vulkan
```

See: [GPU Backends](./TUTORIAL_GPU_BACKENDS.md)

## What's next?

| I want to... | Read |
|--------------|------|
| Understand tensor shapes and slicing | [First Steps](./TUTORIAL_FIRST_STEPS.md), [Slicing](./TUTORIAL_SLICING.md) |
| Learn how broadcasting works | [Broadcasting](./TUTORIAL_BROADCASTING.md) |
| Use map/reduce operations | [Map and Reduce](./TUTORIAL_MAP_REDUCE.md) |
| Understand gradients in depth | [Autograd](./TUTORIAL_AUTOGRAD.md) |
| Build and train models | [Neural Networks](./TUTORIAL_NEURAL_NETWORKS.md) |
| Choose an optimizer and scheduler | [Optimizers](./TUTORIAL_OPTIMIZERS.md) |
| Use advanced LA (QR, LU, SVD, ...) | [Advanced LA](./TUTORIAL_ADVANCED_LA.md) |
| Run on GPU | [GPU Backends](./TUTORIAL_GPU_BACKENDS.md) |
| See full working examples | [`examples/`](../examples/) |