module autograd

import math
import vtl

fn test_sin() {
	f64_ctx := ctx[f64]()

	x := f64_ctx.variable(vtl.from_1d([1.0])!)

	mut f := x.sin()!

	f.backprop()!

	expected_value := math.cos(1.0)
	expected_grad := vtl.from_1d([expected_value])!

	assert x.grad.array_equal(expected_grad)
}

fn test_cos() {
	f64_ctx := ctx[f64]()

	x := f64_ctx.variable(vtl.from_1d([1.0])!)

	mut f := x.cos()!

	f.backprop()!

	expected_value := -math.sin(1.0)
	expected_grad := vtl.from_1d([expected_value])!

	assert x.grad.array_equal(expected_grad)
}

fn test_tan() {
	f64_ctx := ctx[f64]()

	x := f64_ctx.variable(vtl.from_1d([1.0])!)

	mut f := x.tan()!

	f.backprop()!

	expected_value := 1.0 / math.pow(math.cos(1.0), 2.0)
	expected_grad := vtl.from_1d([expected_value])!

	assert x.grad.array_equal(expected_grad)
}
