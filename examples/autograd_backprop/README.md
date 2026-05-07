# autograd_backprop

Demonstrates automatic differentiation (autograd) by computing the gradient of
`x^y` with respect to `x`.

## What it demonstrates

- Creating an autograd `Context` and wrapping tensors as `Variable`s
- Forward computation through an autograd operation (`pow`)
- Calling `backprop()` to compute gradients via reverse-mode autodiff
- Reading the computed gradient from `x.grad`

## Math

For `f(x) = x^y`:

```
f(3) = 3^2 = 9
f'(x) = y * x^(y-1)  →  f'(3) = 2 * 3^1 = 6
```

## How to run

```sh
v run main.v
```

## Expected output

```
Variable(value: [3.0], grad: [0.0])
Variable(value: [2.0], grad: [0.0])
Variable(value: [9.0], grad: [0.0])
[6.0]
```

The gradient `6.0` matches the analytical derivative `d(3^2)/dx = 2*3 = 6`.

## Concepts

| Term | Meaning |
|------|---------|
| `autograd.ctx[f64]()` | Creates a computation graph context |
| `ctx.variable(tensor)` | Wraps a tensor; gradients accumulate in `.grad` |
| `pow.backprop()` | Traverses the graph in reverse, computing gradients |
| `x.grad` | Holds `∂loss/∂x` after backpropagation |

## Related examples

- [`nn_simple_two_layer`](../nn_simple_two_layer/) — uses autograd inside a full neural network
- [`nn_regression_sine`](../nn_regression_sine/) — regression training via autograd
