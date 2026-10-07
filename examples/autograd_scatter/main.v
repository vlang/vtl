module main

import vtl
import vtl.autograd

fn main() {
	main_run() or { panic(err) }
}

fn main_run() ! {
	mut ctx := autograd.ctx[f64]()
	input := ctx.variable(vtl.from_array([1.0, 2.0, 3.0], [1, 3])!)
	updates := ctx.variable(vtl.from_array([10.0, 20.0, 30.0], [1, 3])!)
	indices := vtl.from_array[int]([1, 1, 2], [1, 3])!
	weights := ctx.variable(vtl.from_array([2.0, 3.0, 5.0], [1, 3])!)

	scattered := input.scatter_add(indices, updates, 1)!
	mut objective := scattered.multiply(weights)!
	objective.backprop()!
	println('scattered: ${scattered.value.to_array()}')
	println('source gradient: ${input.grad.to_array()}')
	println('updates gradient: ${updates.grad.to_array()}')
}
