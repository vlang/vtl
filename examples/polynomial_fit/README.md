# Polynomial regression

Fit a degree-2 power-series polynomial to a small deterministic dataset, then
evaluate it at a new point. This is a CPU `f64` example with no external data or
dependencies.

Run from `~/.vmodules`:

```sh
v run ./vtl/examples/polynomial_fit/main.v
```

Expected output:

```text
ascending coefficients: [1.0, 0.0, 1.0]
p(3): 10.0
```

Coefficients are stored in ascending powers: `[1, 0, 1]` represents
`1 + 0*x + 1*x^2`. `polynomial_fit` currently uses a small least-squares
solver; it is not a numerically robust replacement for all `numpy.polyfit`
workloads. See the [NumPy parity tracker](../../docs/NUMPY_PARITY.md#current-strengths)
for the supported subset and numerical caveats.
