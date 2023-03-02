module autograd

import vtl

fn test_pow() {
	f64_ctx := ctx[f64]()

	x := f64_ctx.variable(vtl.from_1d([3.0])!)
	y := f64_ctx.variable(vtl.from_1d([2.0])!)

	mut f := x.pow(y)!

	f.backprop()!

	assert x.grad.array_equal(vtl.from_1d([6.0])!)
}
