module main

import vtl
import vtl.autograd

fn main() {
	main_run() or { panic(err) }
}

fn main_run() ! {
	mut ctx := autograd.ctx[f64]()
	input := ctx.variable(vtl.from_array([10.0, 20.0, 30.0], [1, 3])!)
	updates := ctx.variable(vtl.from_array([100.0, 200.0, 300.0], [1, 3])!)
	indices := vtl.from_array[int]([1, 1, 2], [1, 3])!
	weights := ctx.variable(vtl.from_array([2.0, 3.0, 5.0], [1, 3])!)

	put := input.put_along_axis(indices, updates, 1)!
	mut objective := put.multiply(weights)!
	objective.backprop()!
	println('put: ${put.value.to_array()}')
	println('source gradient: ${input.grad.to_array()}')
	println('updates gradient: ${updates.grad.to_array()}')

	mut flat_ctx := autograd.ctx[f64]()
	flat_input := flat_ctx.variable(vtl.from_array([10.0, 20.0, 30.0, 40.0], [2, 2])!)
	flat_updates := flat_ctx.variable(vtl.from_array([100.0, 200.0], [2])!)
	flat_indices := vtl.from_array[int]([1, 1, 3], [3])!
	flat_weights := flat_ctx.variable(vtl.from_array([2.0, 3.0, 5.0, 7.0], [2, 2])!)
	flat_put := flat_input.put(flat_indices, flat_updates)!
	mut flat_objective := flat_put.multiply(flat_weights)!
	flat_objective.backprop()!
	println('flat put: ${flat_put.value.to_array()}')
	println('flat source gradient: ${flat_input.grad.to_array()}')
	println('flat updates gradient: ${flat_updates.grad.to_array()}')
}
