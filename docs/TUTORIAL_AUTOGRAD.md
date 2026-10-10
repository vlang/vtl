# Tutorial: Automatic Differentiation

VTL includes a reverse-mode automatic differentiation engine in the
`vtl.autograd` module.  It lets you compute gradients of any scalar loss
with respect to any input tensor without writing derivative formulas by hand.

## Core concepts

| Concept | Description |
|---------|-------------|
| `Context` | The computation graph; owns all `Variable`s |
| `Variable` | A tensor node in the graph; holds a value and a gradient |
| Forward pass | Computes the output and records the operation in the graph |
| `backprop()` | Traverses the graph in reverse (chain rule) to fill `.grad` fields |

## Creating variables

```v
import vtl
import vtl.autograd

ctx := autograd.ctx[f64]()

x := ctx.variable(vtl.from_1d([3.0])!)
y := ctx.variable(vtl.from_1d([2.0])!)
```

Variables created from the same `ctx` are connected in the same graph.

## Forward computation

Standard operations on variables build the graph automatically:

```v ignore
// assumes x and y are Variables created above
mut z := x.pow(y)! // z = x^y = 3^2 = 9
```

The `pow` call records that `z` depends on `x` and `y`.

## Backpropagation

Call `backprop()` on the final variable (the loss or output):

```v ignore
// assumes z was computed above
z.backprop()!

println(x.grad) // [6.0]  — because d(x^2)/dx = 2x = 2*3 = 6
println(y.grad) // [9.887...] — because d(x^y)/dy = x^y * ln(x)
```

Use `sum()` or `mean()` to turn a tensor into a one-element loss. Their
backward passes broadcast the scalar gradient to the input; `mean()` divides
each input gradient by the number of elements:

```v
import vtl
import vtl.autograd

mut ctx := autograd.ctx[f64]()
x := ctx.variable(vtl.from_1d([2.0, 3.0])!)
mut loss := x.multiply(x)!.sum()!
loss.backprop()!
assert x.grad.to_array() == [4.0, 6.0]
```

See the [scalar loss example](../examples/autograd_scalar_loss/README.md) for
global and axis reductions. `sum_along_axis(axis, keepdims)` and
`mean_along_axis(axis, keepdims)` also route gradients back along the selected
axis. When `keepdims` is false, the backward pass restores the removed axis
before broadcasting.

`cumsum(axis)` is differentiable as well. Its backward pass accumulates each
output gradient toward the beginning of its axis slice:

```v
import vtl
import vtl.autograd

mut ctx := autograd.ctx[f64]()
x := ctx.variable(vtl.from_array([1.0, 2.0, 3.0, 4.0], [2, 2])!)
mut cumulative := x.cumsum(1)!
cumulative.backprop()!
assert x.grad.to_array() == [2.0, 1.0, 2.0, 1.0]
```

`cumprod(axis)` also supports backpropagation, including inputs with zeros:

```v
import vtl
import vtl.autograd

mut ctx := autograd.ctx[f64]()
x := ctx.variable(vtl.from_1d([2.0, 0.0, 3.0])!)
mut cumulative := x.cumprod(0)!
cumulative.backprop()!
assert x.grad.to_array() == [1.0, 8.0, 0.0]
```

## Gradient accumulation

Gradients accumulate across calls.  Zero them before each training step
(the optimizer handles this automatically in the `Sequential` model).

## Unary element-wise gates

VTL provides differentiable unary element-wise operations via `Variable` methods:

| Method | Forward | Backward |
|--------|---------|----------|
| `.log[T]()!` | `log(x)` | `grad / x` |
| `.abs_op[T]()!` | `|x|` | `grad * sign(x)` |
| `.sqrt_op[T]()!` | `sqrt(x)` | `grad / (2*sqrt(x))` |
| `.tanh_op[T]()!` | `tanh(x)` | `grad * (1 - tanh²(x))` |
| `.clamp[T](min, max)!` | `clamp(x, min, max)` | `grad` where min < x < max, else 0 |

```v
import vtl
import vtl.autograd

mut ctx := autograd.ctx[f64]()
x := ctx.variable(vtl.from_1d[f64]([0.5, 1.0, 2.0])!)

mut y := x.log[f64]()!
y.backprop()!
// x.grad ≈ [2.0, 1.0, 0.5]

mut y2 := x.sqrt_op[f64]()!
y2.backprop()!
// y2.grad ≈ [0.707, 0.5, 0.354]

mut y3 := x.clamp[f64](0.0, 1.5)!
y3.backprop()!
// x.grad = [1.0, 1.0, 0.0]  (gradient flows only for values in [0, 1.5])
```

## Shape gates

Changing the shape of a tensor through a `Variable`:

```v
import vtl
import vtl.autograd

mut ctx := autograd.ctx[f64]()
x := ctx.variable(vtl.from_1d[f64]([1.0, 2.0, 3.0, 4.0])!)
y := x.reshape[f64]([2, 2])!
// y.value = [[1,2],[3,4]]

z := x.transpose_op[f64]([1, 0])!
```

Variables can also be joined along an existing axis with `autograd.concatenate`,
or along a new axis with `autograd.stack`. Both operations route the gradient to
each input during backpropagation:

```v
import vtl
import vtl.autograd

ctx := autograd.ctx[f64]()
left := ctx.variable(vtl.from_1d[f64]([1.0, 2.0])!)
right := ctx.variable(vtl.from_1d[f64]([3.0, 4.0])!)

mut rows := autograd.stack[f64]([left, right], axis: 0)!
// rows.value.shape == [2, 2]
rows.backprop()!
// left.grad and right.grad each have shape [2]
```

For concatenation, inputs must share an autograd context and have equal sizes
on every axis except the concatenation axis:

```v
import vtl
import vtl.autograd

ctx := autograd.ctx[f64]()
left := ctx.variable(vtl.from_1d[f64]([1.0, 2.0])!)
right := ctx.variable(vtl.from_1d[f64]([3.0, 4.0])!)

mut joined := autograd.concatenate[f64]([left, right], axis: 0)!
// joined.value == [1.0, 2.0, 3.0, 4.0]
joined.backprop()!
```

## Supported operations

Only operations with a registered autograd rule are tracked. Common ones used
in neural networks include:

- `add`, `subtract`, `multiply` — element-wise arithmetic
- `matmul` — matrix multiplication (used by `linear` layers)
- `pow` — power function
- activation functions: `relu`, `sigmoid`, `elu`, `leaky_relu`
- loss functions: `mse_loss`, `sigmoid_cross_entropy`, `softmax_cross_entropy`

## Full example

```v
import vtl
import vtl.autograd

ctx := autograd.ctx[f64]()

x := ctx.variable(vtl.from_1d([3.0])!)
y := ctx.variable(vtl.from_1d([2.0])!)

mut result := x.pow(y)!
result.backprop()!

println(result) // Variable(value: [9.0], ...)
println(x.grad) // [6.0]
```

Run the full example: [`examples/autograd_backprop`](../examples/autograd_backprop/).

## Next steps

- [Neural Networks Tutorial](./TUTORIAL_NEURAL_NETWORKS.md) — build and train models with autograd
- [First Steps](./TUTORIAL_FIRST_STEPS.md) — tensor creation and properties
