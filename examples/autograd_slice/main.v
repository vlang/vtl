module main

import vtl
import vtl.autograd

fn main() {
	main_run() or { panic(err) }
}

fn main_run() ! {
	mut ctx := autograd.ctx[f64]()
	input := ctx.variable(vtl.from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2])!)
	mut rows := input.slice([0, 3, 2], []int{})!
	rows.backprop()!
	println('sliced: ${rows.value.to_array()}')
	println('source gradient: ${input.grad.to_array()}')
}
