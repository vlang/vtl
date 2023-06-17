module autograd

import vtl

fn test_add() {
	f64_ctx := ctx[f64]()

	x := f64_ctx.variable(vtl.from_1d([1.0])!)
	y := f64_ctx.variable(vtl.from_1d([1.0])!)

	mut f := x.add(y)!

	f.backprop()!

	expected_grad := vtl.from_1d([1.0])!

	assert x.grad.array_equal(expected_grad)
}

fn test_subtract() {
	f64_ctx := ctx[f64]()

	x := f64_ctx.variable(vtl.from_1d([1.0])!)
	y := f64_ctx.variable(vtl.from_1d([1.0])!)

	mut f := x.subtract(y)!

	f.backprop()!

	expected_grad := vtl.from_1d([1.0])!

	assert x.grad.array_equal(expected_grad)
}

fn test_multiply() {
	f64_ctx := ctx[f64]()

	x := f64_ctx.variable(vtl.from_1d([3.0])!)
	y := f64_ctx.variable(vtl.from_1d([2.0])!)

	mut f := x.multiply(y)!

	f.backprop()!

	expected_grad := vtl.from_1d([2.0])!

	assert x.grad.array_equal(expected_grad)
}

fn test_divide() {
	f64_ctx := ctx[f64]()

	x := f64_ctx.variable(vtl.from_1d([3.0])!)
	y := f64_ctx.variable(vtl.from_1d([2.0])!)

	mut f := x.divide(y)!

	f.backprop()!

	expected_grad := vtl.from_1d([0.5])!

	assert x.grad.array_equal(expected_grad)
}
