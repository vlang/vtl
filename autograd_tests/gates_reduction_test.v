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

fn test_autograd_concat_forward_backward() ! {
	ctx := autograd.ctx[f64]()
	a := ctx.variable(vtl.from_array([1.0, 2], [1, 2])!)
	b := ctx.variable(vtl.from_array([3.0, 4, 5], [1, 3])!)
	mut result := autograd.concat[f64]([a, b], -1)!
	assert result.value.shape == [1, 5]
	assert result.value.to_array() == [1.0, 2, 3, 4, 5]
	result.backprop()!
	assert a.grad.shape == [1, 2]
	assert b.grad.shape == [1, 3]
	assert a.grad.to_array() == [1.0, 1]
	assert b.grad.to_array() == [1.0, 1, 1]
}

fn test_autograd_stack_forward_backward() ! {
	ctx := autograd.ctx[f64]()
	a := ctx.variable(vtl.from_1d([1.0, 2])!)
	b := ctx.variable(vtl.from_1d([3.0, 4])!)
	mut result := autograd.stack[f64]([a, b], -1)!
	assert result.value.shape == [2, 2]
	assert result.value.to_array() == [1.0, 3, 2, 4]
	result.backprop()!
	assert a.grad.shape == [2]
	assert b.grad.shape == [2]
	assert a.grad.to_array() == [1.0, 1]
	assert b.grad.to_array() == [1.0, 1]
}

fn test_reshape_and_transpose_gates_preserve_nonuniform_gradient_values() ! {
	ctx := autograd.ctx[f64]()
	mut reshaped := ctx.variable(vtl.from_array([0.0, 0, 0, 0], [4])!)
	reshaped.grad = vtl.from_array([1.0, 2, 3, 4], [4])!
	reshape_grads := autograd.reshape_gate[f64]([2, 2]).backward(autograd.payload(reshaped))!
	assert reshape_grads[0].shape == [2, 2]
	assert reshape_grads[0].to_array() == [1.0, 2, 3, 4]

	mut transposed := ctx.variable(vtl.from_array([0.0, 0, 0, 0, 0, 0], [3, 2])!)
	transposed.grad = vtl.from_array([1.0, 2, 3, 4, 5, 6], [3, 2])!
	transpose_grads := autograd.transpose_gate[f64]([1, 0]).backward(autograd.payload(transposed))!
	assert transpose_grads[0].shape == [2, 3]
	assert transpose_grads[0].to_array() == [1.0, 3, 5, 2, 4, 6]
}
