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

```v ignore
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

```v ignore
import vtl
import vtl.autograd as ag

mut ctx := ag.ctx[f64]()
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

## Reduction gates

Reduction operations on `Variable`:

```v ignore
import vtl
import vtl.autograd as ag

mut ctx := ag.ctx[f64]()
flat := vtl.from_1d[f64]([1.0, 2.0, 3.0, 4.0])!
x := ctx.variable(flat.reshape([2, 2])!)

// Sum: reduce axis 1 (columns) → [3.0, 7.0]
s := vtl.sum[f64](x.value, 1)!
mut var_s := ctx.variable(s)
var_s.backprop()!
```

## Shape gates

Changing the shape of a tensor through a `Variable`:

```v
import vtl
import vtl.autograd as ag

mut ctx := ag.ctx[f64]()
x := ctx.variable(vtl.from_1d[f64]([1.0, 2.0, 3.0, 4.0])!)
y := x.reshape[f64]([2, 2])!
// y.value = [[1,2],[3,4]]

z := x.transpose_op[f64]([1, 0])!
```

## Supported operations

The autograd engine tracks every VTL tensor operation.  Common ones used
in neural networks:

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

mut pow := x.pow(y)!
pow.backprop()!

println(pow) // Variable(value: [9.0], ...)
println(x.grad) // [6.0]
```

Run the full example: [`examples/autograd_backprop`](../examples/autograd_backprop/).

## Next steps

- [Neural Networks Tutorial](./TUTORIAL_NEURAL_NETWORKS.md) — build and train models with autograd
- [First Steps](./TUTORIAL_FIRST_STEPS.md) — tensor creation and properties
