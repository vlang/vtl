module main

import vtl
import vtl.autograd

fn main() {
	ctx := autograd.new_ctx<f64>()

	x := ctx.variable(vtl.from_1d([3.0])?)
	y := ctx.variable(vtl.from_1d([2.0])?)

	println(x)
	println(y)

	mut f := autograd.pow(x, y)?

	f.backprop()?

	println(f)
	println(x.grad)
}
