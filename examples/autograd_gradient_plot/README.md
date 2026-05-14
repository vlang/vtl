# autograd_gradient_plot

Demonstrates VTL's automatic differentiation by computing `sin(x)` and its
gradient `cos(x)` across [-2π, 2π], then plotting both curves with `vsl.plot`.

## What it demonstrates

- Using `Variable.sin()` for differentiable forward computation
- Calling `backprop()` to compute df/dx automatically
- Verifying autograd accuracy against the analytical derivative
- Overlaying autograd gradient and expected cos(x) on one plot
- Printing a numerical accuracy table at key points

## Run

```sh
v run .
```

An interactive plot opens comparing:
- `f(x) = sin(x)` (blue solid line)
- `f'(x)` via autograd (orange dashed line)
- `cos(x)` expected (green dotted line)

The autograd and expected curves overlap perfectly, confirming correctness.
