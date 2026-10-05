module autograd_tests

import math
import vtl
import vtl.autograd

fn square_f64(input &autograd.Variable[f64]) !&autograd.Variable[f64] {
	return input.multiply(input)!
}

fn sine_f64(input &autograd.Variable[f64]) !&autograd.Variable[f64] {
	return input.sin()!
}

fn vector_square_sum_f64(input &autograd.Variable[f64]) !&autograd.Variable[f64] {
	return input.multiply(input)!
}

fn test_grad_check_scalar_f64() ! {
	context := autograd.ctx[f64]()
	input := context.variable(vtl.from_1d([1.5])!)
	assert autograd.grad_check[f64](input, square_f64, 1e-5, 1e-7)!
	assert input.value.get_nth(0) == 1.5
	assert math.abs(input.grad.get_nth(0) - 3.0) < 1e-12
}

fn test_grad_check_vector_input_f64() ! {
	context := autograd.ctx[f64]()
	input := context.variable(vtl.from_1d([-1.0, 0.5, 2.0])!)
	assert autograd.grad_check[f64](input, vector_square_sum_f64, 1e-5, 1e-7)!
	assert input.value.array_equal(vtl.from_1d([-1.0, 0.5, 2.0])!)
	assert input.grad.array_equal(vtl.from_1d([-2.0, 1.0, 4.0])!)
}

fn test_grad_check_nonlinear_f64() ! {
	context := autograd.ctx[f64]()
	input := context.variable(vtl.from_1d([0.25, 1.0])!)
	assert autograd.grad_check[f64](input, sine_f64, 1e-5, 1e-7)!
	assert math.abs(input.grad.get_nth(0) - math.cos(0.25)) < 1e-12
}

fn test_grad_check_rejects_invalid_epsilon() {
	context := autograd.ctx[f64]()
	input := context.variable(vtl.from_1d([1.5])!)
	_ := autograd.grad_check[f64](input, square_f64, 0.0, 1e-7) or {
		assert err.msg().contains('eps must be positive')
		return
	}
	assert false, 'expected invalid epsilon to return an error'
}

fn test_grad_check_rejects_invalid_tolerance() {
	context := autograd.ctx[f64]()
	input := context.variable(vtl.from_1d([1.0])!)
	_ := autograd.grad_check[f64](input, square_f64, 1e-5, -1.0) or {
		assert err.msg().contains('tolerance must be non-negative')
		return
	}
	assert false, 'expected invalid tolerance to return an error'
}
