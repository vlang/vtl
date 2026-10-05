# VTL Autograd

`vtl.autograd` tracks tensor operations and propagates gradients through the
recorded computation graph. Create related variables from the same context,
build the forward computation, and call `backprop()` on its result.

```v
import vtl
import vtl.autograd

context := autograd.ctx[f64]()
x := context.variable(vtl.from_1d([2.0])!)
mut y := x.multiply(x)!
y.backprop()!
println(x.grad) // [4.0]
```

## Checking gradients numerically

`autograd.grad_check` compares the analytical gradient with central finite
differences. It is useful when implementing a gate or debugging a model. The
context must have no pending graph before calling it. The callback must build a
fresh output from the supplied input each time it runs.

```v
import vtl
import vtl.autograd

fn square(input &autograd.Variable[f64]) !&autograd.Variable[f64] {
	return input.multiply(input)!
}

context := autograd.ctx[f64]()
mut x := context.variable(vtl.from_1d([2.0])!)
matches := autograd.grad_check[f64](mut x, square, 1e-5, 1e-6)!
assert matches
assert x.grad.get_nth(0) == 4.0
```

The callback may return a tensor of any non-zero size. The checker compares the
gradient of the sum of its output elements, matching `backprop()`'s all-ones
initial gradient. `eps` must be positive. `tolerance` is a non-negative
relative tolerance with a unit absolute scale near zero. The input value is
restored after checking; its gradient is replaced by the analytical gradient.

Call the V compiler from the parent of the `vtl` clone:

```sh
systemd-run --user --scope --quiet --property=MemoryMax=1G --setenv=VJOBS=2 \
	--working-directory="$HOME/.vmodules" -- v test ./vtl/autograd_tests
```

`backprop()` seeds the output gradient with ones and accumulates gradients in
its ancestors. It consumes the recorded graph, so run a fresh forward pass for
each subsequent backward pass. Set `requires_grad: false` when creating a
variable that should not receive gradients. See the
[autograd tutorial](../docs/TUTORIAL_AUTOGRAD.md) and
[backprop example](../examples/autograd_backprop/README.md).
The module direction and planned work are summarized in the
[VTL roadmap](../ROADMAP.md).

The graph records operations that have an autograd gate; it does not promise to
track every tensor API operation. Check the relevant operation and tests before
depending on its gradient behavior. GPU autograd paths are backend-specific and
experimental; see [device memory notes](../docs/DEVICE_MEMORY.md).

## Tracked operations

`Variable` methods record gates for elementwise add, subtract, multiply,
divide, and power; exponential, logarithm, sine, cosine, tangent, absolute
value, square root, hyperbolic tangent, and clamp; matrix multiplication; sum
and mean reductions; reshape, permutation transpose, and concatenation. See
[`variable_ops.v`](variable_ops.v) for method signatures and the
[`gates` reference](../nn/gates/README.md) for the backward-rule adapters.

The implementation is grouped by rule in `gates_basic.v` (arithmetic),
`gates_pow.v`, `gates_exp.v`, `gates_trig.v`, `gates_unary.v`,
`gates_blas.v`, and `gates_reduction.v`. This list describes registered
autograd rules, not every operation available on `Tensor`.

CUDA-specific variables and helpers are conditional builds. Accelerator
coverage depends on the operation and backend; a GPU tensor does not imply
that every forward operation or backward rule stays device-resident. See
[`DEVICE_MEMORY.md`](../docs/DEVICE_MEMORY.md) for the current caveats.
