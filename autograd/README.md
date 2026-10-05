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
x := context.variable(vtl.from_1d([2.0])!)
matches := autograd.grad_check[f64](x, square, 1e-5, 1e-6)!
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
