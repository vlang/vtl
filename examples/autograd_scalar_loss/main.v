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
}
