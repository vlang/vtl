module autograd_tests

import vtl
import vtl.autograd

fn test_remainder_backward_matches_pytorch_convention() ! {
	context := autograd.ctx[f64]()
	a := context.variable(vtl.from_1d([-5.0, 5.0])!)
	b := context.variable(vtl.from_1d([3.0, -3.0])!)

	mut result := a.remainder(b)!
	assert result.value.to_array() == [1.0, -1.0]
	result.backprop()!

	assert a.grad.to_array() == [1.0, 1.0]
	assert b.grad.to_array() == [2.0, 2.0]
}

fn test_remainder_backward_reduces_broadcast_divisor_gradient() ! {
	context := autograd.ctx[f64]()
	a := context.variable(vtl.from_array([5.0, 7.0, 9.0, 11.0], [2, 2])!)
	b := context.variable(vtl.from_1d([2.0, 3.0])!)

	mut result := a.remainder(b)!
	result.backprop()!

	assert a.grad.to_array() == [1.0, 1.0, 1.0, 1.0]
	assert b.grad.to_array() == [-6.0, -5.0]
}

fn test_remainder_backward_reduces_scalar_divisor_gradient() ! {
	context := autograd.ctx[f64]()
	a := context.variable(vtl.from_1d([5.0, 7.0])!)
	b := context.variable(vtl.from_1d([2.0])!)

	mut result := a.remainder(b)!
	result.backprop()!

	assert a.grad.to_array() == [1.0, 1.0]
	assert b.grad.to_array() == [-5.0]
}

fn test_remainder_backward_reduces_broadcast_left_gradient() ! {
	context := autograd.ctx[f64]()
	a := context.variable(vtl.from_1d([5.0])!)
	b := context.variable(vtl.from_1d([2.0, 3.0])!)

	mut result := a.remainder(b)!
	result.backprop()!

	assert a.grad.to_array() == [2.0]
	assert b.grad.to_array() == [-2.0, -1.0]
}

fn test_remainder_backward_restores_zero_rank_scalar_shape() ! {
	context := autograd.ctx[f64]()
	a := context.variable(vtl.from_array([5.0], []int{})!)
	b := context.variable(vtl.from_1d([2.0, 3.0])!)

	mut result := a.remainder(b)!
	assert result.value.to_array() == [1.0, 2.0]
	result.backprop()!

	assert a.grad.shape.len == 0
	assert a.grad.get_nth(0) == 2.0
	assert b.grad.to_array() == [-2.0, -1.0]
}

fn test_remainder_backward_uses_floor_quotient_at_exact_multiples() ! {
	context := autograd.ctx[f64]()
	a := context.variable(vtl.from_1d([6.0, -6.0])!)
	b := context.variable(vtl.from_1d([3.0, -3.0])!)

	mut result := a.remainder(b)!
	assert result.value.get_nth(0) == 0.0
	assert result.value.get_nth(1) == 0.0
	result.backprop()!

	assert a.grad.to_array() == [1.0, 1.0]
	assert b.grad.to_array() == [-2.0, -2.0]
}

fn test_remainder_backward_supports_f32() ! {
	context := autograd.ctx[f32]()
	a := context.variable(vtl.from_1d[f32]([5.0])!)
	b := context.variable(vtl.from_1d[f32]([2.0])!)

	mut result := a.remainder(b)!
	assert result.value.get_nth(0) == f32(1.0)
	result.backprop()!

	assert a.grad.get_nth(0) == f32(1.0)
	assert b.grad.get_nth(0) == f32(-2.0)
}

fn test_remainder_backward_propagates_ieee_non_finite_values() ! {
	context := autograd.ctx[f64]()
	zero := vtl.from_1d([0.0])!
	infinity := vtl.from_1d([1.0])!.divide(zero)!.get_nth(0)
	not_a_number := zero.divide(zero)!.get_nth(0)
	a := context.variable(vtl.from_1d([4.0, infinity, not_a_number])!)
	b := context.variable(vtl.from_1d([2.0, 2.0, 2.0])!)

	mut result := a.remainder(b)!
	assert result.value.get_nth(0) == 0.0
	assert result.value.is_nan().get_nth(1)
	assert result.value.is_nan().get_nth(2)
	result.backprop()!

	assert a.grad.to_array() == [1.0, 1.0, 1.0]
	assert b.grad.get_nth(0) == -2.0
	assert b.grad.is_inf(0).get_nth(1)
	assert b.grad.get_nth(1) < 0.0
	assert b.grad.is_nan().get_nth(2)
}

fn remainder_grad_check_by_two(input &autograd.Variable[f64]) !&autograd.Variable[f64] {
	divisor := input.context.variable(vtl.from_1d([2.0])!, requires_grad: false)
	return input.remainder(divisor)!
}

fn test_remainder_backward_matches_finite_difference_away_from_discontinuities() ! {
	context := autograd.ctx[f64]()
	mut input := context.variable(vtl.from_1d([1.25, 3.5])!)

	assert autograd.grad_check[f64](mut input, remainder_grad_check_by_two, 1e-5, 1e-7)!
	assert input.grad.to_array() == [1.0, 1.0]
}

fn test_integer_remainder_rejects_autograd() {
	context := autograd.ctx[int]()
	a := context.variable(vtl.from_1d([5, 7])!)
	b := context.variable(vtl.from_1d([2, 3])!)
	_ := a.remainder(b) or {
		assert err.msg().contains('requires f32 or f64')
		return
	}
	assert false, 'expected integer Variable remainder to reject autograd'
}
