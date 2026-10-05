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
	value := vtl.concatenate[f64]([x.value, y.value], axis: 0)!
	mut result := f64_ctx.variable(value)
	gate := autograd.concat_gate[f64](0, [2, 3])
	gate.cache(mut result, x, y)!
	result.backprop()!
	assert x.grad.shape == [2]
	assert y.grad.shape == [3]
	assert x.grad.get_nth(0) == f64(1)
	assert x.grad.get_nth(1) == f64(1)
	assert y.grad.get_nth(0) == f64(1)
	assert y.grad.get_nth(1) == f64(1)
	assert y.grad.get_nth(2) == f64(1)
}
