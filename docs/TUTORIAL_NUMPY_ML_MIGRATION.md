# NumPy and classical ML workflows in VTL

This guide connects familiar NumPy operations to runnable VTL examples. The
examples use CPU execution and explicit `f64` data unless noted; they are small
learning workflows, not broad performance or framework-parity claims. Run the
commands from `~/.vmodules`.

## Prepare data and fit a classical model

`numpy.ndarray` maps to a VTL `Tensor[T]` with an explicit shape. The
[`polynomial_fit` example](../examples/polynomial_fit/main.v) prepares a small
`f64` dataset, fits a degree-2 polynomial, and evaluates the fitted model:

```sh
v run ./vtl/examples/polynomial_fit/main.v
```

```text
ascending coefficients: [1.0, approximately 0.0, 1.0]
p(3): 10.0
ascending Chebyshev coefficients: [2.375, approximately 1.5, 1.125]
Chebyshev p(3): 10.0
```

The corresponding APIs are `vtl.from_array`, `vtl.polynomial_fit`, and
`Polynomial.eval`. Unlike NumPy's broader polynomial interface, VTL currently
supports a documented subset. Fitting uses a domain-scaled Vandermonde matrix
and Householder QR, then returns coefficients in the original power basis;
conversion to that basis can still amplify errors on ill-conditioned data. See the
[polynomial API notes](./NUMPY_PARITY.md#current-strengths).

`vtl.Chebyshev` stores first-kind coefficients in ascending order and supports
scalar/tensor evaluation, arithmetic, calculus, roots up to degree two, and
conversion to/from power series. Supply an explicit `domain` and `window`;
evaluation maps the input domain linearly into the window. Polynomial APIs
currently use `f64`. `vtl.chebyshev_fit` fits in that mapped coordinate and
returns the Chebyshev coefficients. The runnable fitting example demonstrates
both bases.

## Transform samples and preserve gradients

For pure tensor callbacks, use `vtl.vmap`; for callbacks that must participate in
reverse-mode autograd, use `vtl.autograd.vmap`. The
[migration example](../examples/pytorch_migration/main.v) executes a scalar loss
and gradient, then maps a differentiable callback:

```sh
v run ./vtl/examples/pytorch_migration/main.v
```

```text
loss: 14.0
gradient: [2.0, 4.0, 6.0]
vmap: [2.0, 4.0, 6.0]
```

The [functional transform tutorial](./TUTORIAL_FUNCTIONAL_TRANSFORMS.md)
documents axis insertion, output-shape constraints, and current limitations,
including CPU-only execution and no transformation of module state or
randomness.

## Train and evaluate mini-batches

The [`nn_multiclass_iris` example](../examples/nn_multiclass_iris/main.v) creates
a deterministic synthetic three-class dataset, batches it, trains an `f64`
`Sequential` model, and evaluates training accuracy:

```sh
v run ./vtl/examples/nn_multiclass_iris/main.v
```

With its fixed seed, the current result is loss `0.039860` at epoch 499 and
training accuracy `60/60`. This evaluates the same samples used for training;
it is not a held-out score. See its
[README](../examples/nn_multiclass_iris/README.md) for the batch layout and
training parameters.

## Save a checkpoint and run inference

The
[`sequential_inference_checkpoint` example](../examples/sequential_inference_checkpoint.v)
trains a CPU `f64` model, saves weights, builds the matching architecture in a
fresh context, reloads the checkpoint, and performs inference:

```sh
v run ./vtl/examples/sequential_inference_checkpoint.v
```

With seed 42, the example currently prints:

```text
prediction: &[[1.4322641669000282]]
```

The numeric value may change with the V/runtime random-number implementation.
The checkpoint format currently supports a limited CPU `Sequential`
architecture and is a weights file, not a portable executable or a general
deployment package. The
[checkpoint guide](./TUTORIAL_INFERENCE_CHECKPOINT.md) defines validation and
unsupported cases.

## Migration boundaries

| NumPy / ML pattern | VTL path | Important difference |
|---|---|---|
| `ndarray` construction | `vtl.from_array(data, shape)` | Choose `f32` or `f64` explicitly; shape is explicit. |
| Polynomial fitting/evaluation | `vtl.polynomial_fit`, `vtl.chebyshev_fit`, and `Polynomial.eval` | Power series and first-kind Chebyshev are supported; fitting uses scaled QR, but conversion to original power coefficients can still amplify conditioning errors. |
| Per-sample mapping | `vtl.vmap` or `vtl.autograd.vmap` | CPU-only today; callback output shape must be consistent. |
| Mini-batch training | `Sequential`, losses, optimizers, and tensor slicing | Backend and gradient coverage are operation-specific. |
| Inference checkpoint | `save_inference_checkpoint` / `load_inference_checkpoint` | CPU-only supported layer set; does not include optimizer state or package an executable. |

Use the [NumPy parity tracker](./NUMPY_PARITY.md) and
[PyTorch parity tracker](./PYTORCH_PARITY.md) for current operation, dtype,
device, and backend status. Do not infer general NumPy/PyTorch compatibility
from these focused examples.
