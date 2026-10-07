module main

import vtl
import vtl.autograd

fn main() {
	main_run() or { panic(err) }
}

fn main_run() ! {
	mut sum_context := autograd.ctx[f64]()
	sum_input := sum_context.variable(vtl.from_1d([2.0, 3.0])!)
	mut sum_loss := sum_input.multiply(sum_input)!.sum()!
	sum_loss.backprop()!
	println('sum loss: ${sum_loss.value.to_array()}')
	println('sum gradient: ${sum_input.grad.to_array()}')

	mut mean_context := autograd.ctx[f64]()
	mean_input := mean_context.variable(vtl.from_1d([2.0, 3.0])!)
	mut mean_loss := mean_input.multiply(mean_input)!.mean()!
	mean_loss.backprop()!
	println('mean loss: ${mean_loss.value.to_array()}')
	println('mean gradient: ${mean_input.grad.to_array()}')

	mut axis_sum_context := autograd.ctx[f64]()
	axis_sum_input := axis_sum_context.variable(vtl.from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [
		2,
		3,
	])!)
	mut row_sums := axis_sum_input.sum_along_axis(-1, false)!
	row_sums.backprop()!
	println('row sums: ${row_sums.value.to_array()}')
	println('axis sum gradient: ${axis_sum_input.grad.to_array()}')

	mut axis_mean_context := autograd.ctx[f64]()
	axis_mean_input := axis_mean_context.variable(vtl.from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [
		2,
		3,
	])!)
	mut column_means := axis_mean_input.mean_along_axis(0, true)!
	column_means.backprop()!
	println('column means: ${column_means.value.to_array()}')
	println('axis mean gradient: ${axis_mean_input.grad.to_array()}')
}
