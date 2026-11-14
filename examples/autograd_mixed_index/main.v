module main

import vtl
import vtl.autograd

fn main() {
	main_run() or { panic(err) }
}

fn main_run() ! {
	mut ctx := autograd.ctx[f64]()
	input := ctx.variable(vtl.from_array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [2, 3])!)
	rows := vtl.from_1d([1, 0, 1])!
	columns := vtl.from_1d([2, 0, 2])!
	mut selected := input.mixed_index([vtl.array_index(rows), vtl.array_index(columns)])!
	selected.backprop()!

	println('selected: ${selected.value.to_array()}')
	println('source gradient: ${input.grad.to_array()}')
}
