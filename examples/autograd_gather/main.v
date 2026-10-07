module main

import vtl
import vtl.autograd

fn main() {
	main_run() or { panic(err) }
}

fn main_run() ! {
	mut ctx := autograd.ctx[f64]()
	input := ctx.variable(vtl.from_array([10.0, 20.0, 30.0], [1, 3])!)
	indices := vtl.from_array[int]([1, 1, 2], [1, 3])!
	mut selected := input.take_along_axis(indices, 1)!
	selected.backprop()!
	println('selected: ${selected.value.to_array()}')
	println('source gradient: ${input.grad.to_array()}')
}
