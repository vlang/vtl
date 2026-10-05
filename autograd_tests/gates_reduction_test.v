module autograd_tests

import vtl.autograd
import vtl

fn test_reshape_forward_backward() {
	f64_ctx := autograd.ctx[f64]()
	x := f64_ctx.variable(vtl.from_1d([1.0, 2.0, 3.0, 4.0])!)
	mut f := x.reshape([2, 2])!
	assert f.value.shape == [2, 2]
	f.backprop()!
	// gradient of reshape is reshape back to original
	assert x.grad.shape == [4]
}

fn test_transpose_forward_backward() {
	f64_ctx := autograd.ctx[f64]()
	x := f64_ctx.variable(vtl.from_2d([[1.0, 2.0], [3.0, 4.0]])!)
	mut f := x.transpose_op([1, 0])!
	// transposed: [[1,3],[2,4]]
	assert f.value.shape == [2, 2]
	assert f.value.get_nth(0) == f64(1)
	assert f.value.get_nth(1) == f64(3)
	f.backprop()!
	assert x.grad.shape == [2, 2]
}

fn test_concat_gate_backward_splits_gradient() {
	f64_ctx := autograd.ctx[f64]()
	x := f64_ctx.variable(vtl.from_1d([1.0, 2.0])!)
	y := f64_ctx.variable(vtl.from_1d([3.0, 4.0, 5.0])!)
	mut result := autograd.concatenate[f64]([x, y], axis: 0)!
	result.backprop()!
	assert x.grad.shape == [2]
	assert y.grad.shape == [3]
	assert x.grad.get_nth(0) == f64(1)
	assert x.grad.get_nth(1) == f64(1)
	assert y.grad.get_nth(0) == f64(1)
	assert y.grad.get_nth(1) == f64(1)
	assert y.grad.get_nth(2) == f64(1)
}

fn test_stack_variables_backward_unstacks_gradient() {
	f64_ctx := autograd.ctx[f64]()
	x := f64_ctx.variable(vtl.from_1d([1.0, 2.0])!)
	y := f64_ctx.variable(vtl.from_1d([3.0, 4.0])!)
	mut result := autograd.stack[f64]([x, y], axis: 0)!
	assert result.value.shape == [2, 2]
	result.backprop()!
	assert x.grad.shape == [2]
	assert y.grad.shape == [2]
	assert x.grad.get_nth(0) == f64(1)
	assert x.grad.get_nth(1) == f64(1)
	assert y.grad.get_nth(0) == f64(1)
	assert y.grad.get_nth(1) == f64(1)
}

fn test_stack_variables_without_grad_tracking() {
	f64_ctx := autograd.ctx[f64]()
	x := f64_ctx.variable(vtl.from_1d([1.0, 2.0])!, requires_grad: false)
	y := f64_ctx.variable(vtl.from_1d([3.0, 4.0])!, requires_grad: false)
	result := autograd.stack[f64]([x, y], axis: 0)!
	assert result.value.shape == [2, 2]
	assert !result.requires_grad
}
