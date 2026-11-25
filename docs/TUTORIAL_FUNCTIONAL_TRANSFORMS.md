# Functional tensor transforms

VTL provides `vtl.vmap` for pure tensor functions and
`vtl.autograd.vmap` for functions that must remain differentiable. Both APIs
map one input axis, remove that axis while calling the function, and insert a
new output axis before concatenating the per-sample results.

```v
import vtl
import vtl.stats

fn main() {
	input := vtl.from_array([1.0, 2.0, 3.0, 4.0], [2, 2])!
	output := vtl.vmap[f64](fn (sample &vtl.Tensor[f64]) !&vtl.Tensor[f64] {
		return sample.add(sample)
	}, input, 0, 0)!
	println(output)
}
```

`in_dim` and `out_dim` accept negative axes. The mapped function must return
the same shape for every sample. Scalar reductions can be represented as a
one-element tensor:

```v
output := vtl.vmap[f64](fn (sample &vtl.Tensor[f64]) !&vtl.Tensor[f64] {
	return vtl.from_1d([vtl.stats.sum[f64](sample)])
}, input, 0, 0)!
```

For autograd, use the matching namespace and a variable callback:

```v
import vtl
import vtl.autograd

ctx := vtl.autograd.ctx[f64]()
input := ctx.variable(vtl.from_array([1.0, 2.0, 3.0], [3])!, requires_grad: true)
output := vtl.autograd.vmap[f64](fn (sample &vtl.autograd.Variable[f64]) !&vtl.autograd.Variable[f64] {
	return sample.add(sample)
}, input, 0, 0)!
loss := output.sum()!
loss.backprop()!
```

The current implementation is CPU-only and does not transform module state or
randomness. Empty mapped axes and inconsistent callback shapes are rejected
explicitly rather than producing an ambiguous output shape.
