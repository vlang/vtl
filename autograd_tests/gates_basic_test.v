module autograd_tests

import vtl.autograd
import vtl

fn test_add() {
	f64_ctx := autograd.ctx[f64]()

	x := f64_ctx.variable(vtl.from_1d([1.0])!)
	y := f64_ctx.variable(vtl.from_1d([1.0])!)

	mut f := x.add(y)!

	f.backprop()!

	expected_grad := vtl.from_1d([1.0])!

	assert x.grad.array_equal(expected_grad)
}

fn test_subtract() {
	f64_ctx := autograd.ctx[f64]()

	x := f64_ctx.variable(vtl.from_1d([1.0])!)
	y := f64_ctx.variable(vtl.from_1d([1.0])!)

	mut f := x.subtract(y)!

	f.backprop()!

	expected_grad := vtl.from_1d([1.0])!

	assert x.grad.array_equal(expected_grad)
}

fn test_multiply() {
	f64_ctx := autograd.ctx[f64]()

	x := f64_ctx.variable(vtl.from_1d([3.0])!)
	y := f64_ctx.variable(vtl.from_1d([2.0])!)

	mut f := x.multiply(y)!

	f.backprop()!

	expected_grad := vtl.from_1d([2.0])!

	assert x.grad.array_equal(expected_grad)
	assert y.grad.get_nth(0) == f64(3.0)
}

fn test_multiply_fused_backward_for_f32() {
	ctx := autograd.ctx[f32]()
	x := ctx.variable(vtl.from_1d[f32]([2.0, 4.0])!)
	y := ctx.variable(vtl.from_1d[f32]([3.0, 5.0])!)

	mut result := x.multiply(y)!
	result.backprop()!

	assert x.grad.get_nth(0) == f32(3.0)
	assert x.grad.get_nth(1) == f32(5.0)
	assert y.grad.get_nth(0) == f32(2.0)
	assert y.grad.get_nth(1) == f32(4.0)
}

fn test_multiply_backward_falls_back_for_non_contiguous_inputs() {
	ctx := autograd.ctx[f64]()
	a := ctx.variable(vtl.from_array[f64]([1.0, 3.0, 2.0, 4.0], [2, 2])!.transpose([1, 0])!)
	b := ctx.variable(vtl.from_array[f64]([5.0, 7.0, 6.0, 8.0], [2, 2])!.transpose([1, 0])!)

	mut result := a.multiply(b)!
	result.backprop()!

	assert a.grad.get([0, 0]) == 5.0
	assert a.grad.get([0, 1]) == 6.0
	assert a.grad.get([1, 0]) == 7.0
	assert a.grad.get([1, 1]) == 8.0
	assert b.grad.get([0, 0]) == 1.0
	assert b.grad.get([0, 1]) == 2.0
	assert b.grad.get([1, 0]) == 3.0
	assert b.grad.get([1, 1]) == 4.0
}

fn test_divide() {
	f64_ctx := autograd.ctx[f64]()

	x := f64_ctx.variable(vtl.from_1d([3.0])!)
	y := f64_ctx.variable(vtl.from_1d([2.0])!)

	mut f := x.divide(y)!

	f.backprop()!

	expected_grad := vtl.from_1d([0.5])!

	assert x.grad.array_equal(expected_grad)
	assert y.grad.get_nth(0) == f64(-0.75)
}

fn test_divide_backward_uses_squared_denominator() {
	ctx := autograd.ctx[f64]()
	x := ctx.variable(vtl.from_1d([8.0])!)
	y := ctx.variable(vtl.from_1d([4.0])!)

	mut result := x.divide(y)!
	result.backprop()!

	assert x.grad.get_nth(0) == f64(0.25)
	assert y.grad.get_nth(0) == f64(-0.5)
}

fn test_backprop_accumulates_branch_gradients_in_existing_buffer() {
	ctx := autograd.ctx[f64]()
	x := ctx.variable(vtl.from_1d([2.0])!)
	initial_grad := x.grad

	mut loss := x.multiply(x)!.add(x)!
	loss.backprop()!

	expected_grad := vtl.from_1d([5.0])!
	assert x.grad.array_equal(expected_grad)
	assert voidptr(x.grad) == voidptr(initial_grad)
}

fn test_register_reuses_fresh_zero_gradient_buffer() {
	ctx := autograd.ctx[f64]()
	left := ctx.variable(vtl.from_1d([1.0])!)
	right := ctx.variable(vtl.from_1d([2.0])!)
	mut result := ctx.variable(vtl.from_1d([3.0])!)
	initial_grad := result.grad
	gate := autograd.add_gate[f64]()

	gate.cache(mut result, left, right)!

	assert voidptr(result.grad) == voidptr(initial_grad)
	assert result.grad.get_nth(0) == f64(0.0)
}

fn test_register_allocates_gradient_for_untracked_result() {
	ctx := autograd.ctx[f64]()
	left := ctx.variable(vtl.from_1d([1.0])!)
	right := ctx.variable(vtl.from_1d([2.0])!)
	value := vtl.from_1d([3.0])!
	mut result := ctx.variable(value, requires_grad: false)
	gate := autograd.add_gate[f64]()

	gate.cache(mut result, left, right)!

	assert result.requires_grad
	assert voidptr(result.grad) != voidptr(value)
	assert result.grad.get_nth(0) == f64(0.0)
}
