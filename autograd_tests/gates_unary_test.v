module autograd_tests
import vtl.autograd

import vtl

fn test_log_forward_backward() {
	f64_ctx := autograd.ctx[f64]()
	x := f64_ctx.variable(vtl.from_1d([1.0, 2.0])!)
	mut f := x.log()!
	// forward: log([1, 2]) = [0, ln(2)]
	assert f.value.get_nth(0) == f64(0)
	f64_ln2 := 0.6931471805599453
	diff := f.value.get_nth(1) - f64_ln2
	assert diff * diff < 1e-10
	// backward: d/dx log(x) = 1/x
	f.backprop()!
	assert x.grad.get_nth(0) == f64(1) // 1/1
	half_diff := x.grad.get_nth(1) - 0.5 // 1/2
	assert half_diff * half_diff < 1e-10
}

fn test_abs_forward_backward() {
	f64_ctx := autograd.ctx[f64]()
	x := f64_ctx.variable(vtl.from_1d([-3.0, 4.0])!)
	mut f := x.abs_op()!
	// forward: abs([-3, 4]) = [3, 4]
	assert f.value.get_nth(0) == f64(3)
	assert f.value.get_nth(1) == f64(4)
	// backward: sign(x)
	f.backprop()!
	assert x.grad.get_nth(0) < 0 // sign(-3) = -1
	assert x.grad.get_nth(1) > 0 // sign(4) = 1
}

fn test_sqrt_forward_backward() {
	f64_ctx := autograd.ctx[f64]()
	x := f64_ctx.variable(vtl.from_1d([4.0, 9.0])!)
	mut f := x.sqrt_op()!
	// forward: sqrt([4, 9]) = [2, 3]
	assert f.value.get_nth(0) == f64(2)
	assert f.value.get_nth(1) == f64(3)
	// backward: d/dx sqrt(x) = 0.5 / sqrt(x) * grad
	// grad is all 1s, so d = [0.5/2, 0.5/3] = [0.25, 0.1667]
	f.backprop()!
	d0 := x.grad.get_nth(0) - 0.25
	assert d0 * d0 < 1e-10
}

fn test_tanh_forward_backward() {
	f64_ctx := autograd.ctx[f64]()
	x := f64_ctx.variable(vtl.from_1d([0.0])!)
	mut f := x.tanh_op()!
	// tanh(0) = 0
	assert f.value.get_nth(0) == f64(0)
	// d/dx tanh(x) at x=0: 1 - tanh(0)^2 = 1
	f.backprop()!
	assert x.grad.get_nth(0) == f64(1)
}

fn test_clamp_forward_backward() {
	f64_ctx := autograd.ctx[f64]()
	x := f64_ctx.variable(vtl.from_1d([-2.0, 0.5, 3.0])!)
	mut f := x.clamp(f64(-1), f64(1))!
	// clamp to [-1, 1]: [-1, 0.5, 1]
	assert f.value.get_nth(0) == f64(-1)
	assert f.value.get_nth(1) == f64(0.5)
	assert f.value.get_nth(2) == f64(1)
	// backward: gradient passes through where within bounds, else 0
	f.backprop()!
	assert x.grad.get_nth(0) == f64(0) // clamped, no grad
	assert x.grad.get_nth(1) == f64(1) // within bounds
	assert x.grad.get_nth(2) == f64(0) // clamped, no grad
}
