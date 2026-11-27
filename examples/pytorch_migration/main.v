module main

import vtl
import vtl.autograd

fn main() {
	ctx := autograd.ctx[f64]()
	input := ctx.variable(vtl.from_array([1.0, 2.0, 3.0], [3])!, requires_grad: true)
	loss := input.multiply(input)!.sum()!
	loss.backprop()!
	println('loss: ${loss.value.get_nth(0):.1f}')
	println('gradient: ${input.grad.to_array()}')

	batched := autograd.vmap[f64](fn (sample &autograd.Variable[f64]) !&autograd.Variable[f64] {
		return sample.add(sample)
	}, input, 0, 0)!
	println('vmap: ${batched.value.to_array()}')
}
