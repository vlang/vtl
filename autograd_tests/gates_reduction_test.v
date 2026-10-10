module autograd_tests

import vtl.autograd
import vtl

fn test_variable_sum_and_mean_forward_backward() ! {
	mut context := autograd.ctx[f64]()
	input := context.variable(vtl.from_array([1.0, 2.0, 3.0, 4.0], [2, 2])!)
	mut total := input.sum()!
	assert total.value.shape == [1]
	assert total.value.to_array() == [10.0]
	total.backprop()!
	assert input.grad.to_array() == [1.0, 1.0, 1.0, 1.0]

	mut mean_input := context.variable(vtl.from_1d([1.0, 2.0, 3.0, 4.0])!)
	mut average := mean_input.mean()!
	assert average.value.to_array() == [2.5]
	average.backprop()!
	assert mean_input.grad.to_array() == [0.25, 0.25, 0.25, 0.25]
}

fn test_variable_sum_backpropagates_through_computation() ! {
	mut context := autograd.ctx[f64]()
	input := context.variable(vtl.from_1d([2.0, 3.0])!)
	squared := input.multiply(input)!
	mut loss := squared.sum()!
	loss.backprop()!
	assert input.grad.to_array() == [4.0, 6.0]
}

fn test_variable_sum_and_mean_support_zero_dimensional_inputs() ! {
	mut context := autograd.ctx[f64]()
	input := context.variable(vtl.from_array([5.0], []int{})!)
	mut total := input.sum()!
	assert total.value.to_array() == [5.0]
	total.backprop()!
	assert input.grad.shape.len == 0
	assert input.grad.get_nth(0) == 1.0

	mut mean_context := autograd.ctx[f64]()
	mean_input := mean_context.variable(vtl.from_array([5.0], []int{})!)
	mut average := mean_input.mean()!
	assert average.value.to_array() == [5.0]
	average.backprop()!
	assert mean_input.grad.shape.len == 0
	assert mean_input.grad.get_nth(0) == 1.0
}

fn test_variable_axis_sum_and_mean_forward_backward() ! {
	mut sum_context := autograd.ctx[f64]()
	sum_input := sum_context.variable(vtl.from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [
		2,
		3,
	])!)
	mut row_sums := sum_input.sum_along_axis(-1, false)!
	assert row_sums.value.shape == [2]
	assert row_sums.value.to_array() == [6.0, 15.0]
	row_sums.backprop()!
	assert sum_input.grad.to_array() == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

	mut mean_context := autograd.ctx[f64]()
	mean_input := mean_context.variable(vtl.from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [
		2,
		3,
	])!)
	mut column_means := mean_input.mean_along_axis(0, true)!
	assert column_means.value.shape == [1, 3]
	assert column_means.value.to_array() == [2.5, 3.5, 4.5]
	column_means.backprop()!
	assert mean_input.grad.to_array() == [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
}

fn test_variable_cumsum_forward_backward() ! {
	mut context := autograd.ctx[f64]()
	input := context.variable(vtl.from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [
		2,
		3,
	])!)
	mut row_cumulative := input.cumsum(-1)!
	assert row_cumulative.value.to_array() == [1.0, 3.0, 6.0, 4.0, 9.0, 15.0]
	row_cumulative.backprop()!
	assert input.grad.to_array() == [3.0, 2.0, 1.0, 3.0, 2.0, 1.0]

	mut column_context := autograd.ctx[f64]()
	column_input := column_context.variable(vtl.from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [
		2,
		3,
	])!)
	mut column_cumulative := column_input.cumsum(0)!
	assert column_cumulative.value.to_array() == [1.0, 2.0, 3.0, 5.0, 7.0, 9.0]
	column_cumulative.backprop()!
	assert column_input.grad.to_array() == [2.0, 2.0, 2.0, 1.0, 1.0, 1.0]
}

fn test_variable_cumsum_rejects_scalar_and_invalid_axis() {
	context := autograd.ctx[f64]()
	scalar := context.variable(vtl.from_array([1.0], []int{})!)
	if _ := scalar.cumsum(0) {
		assert false, 'cumsum must reject scalar variables'
	}
	vector := context.variable(vtl.from_1d([1.0, 2.0])!)
	if _ := vector.cumsum(1) {
		assert false, 'cumsum must reject out-of-range axes'
	}
}

fn test_variable_cumprod_forward_backward_with_zero_inputs() ! {
	mut context := autograd.ctx[f64]()
	input := context.variable(vtl.from_1d([2.0, 0.0, 3.0])!)
	mut cumulative := input.cumprod(-1)!
	assert cumulative.value.to_array() == [2.0, 0.0, 0.0]
	cumulative.backprop()!
	assert input.grad.to_array() == [1.0, 8.0, 0.0]

	mut matrix_context := autograd.ctx[f64]()
	matrix_input := matrix_context.variable(vtl.from_array([2.0, 3.0, 4.0, 5.0], [2, 2])!)
	mut row_products := matrix_input.cumprod(1)!
	assert row_products.value.to_array() == [2.0, 6.0, 4.0, 20.0]
	row_products.backprop()!
	assert matrix_input.grad.to_array() == [4.0, 2.0, 6.0, 4.0]
}

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

fn test_autograd_concatenate_rejects_axis_out_of_range() {
	f64_ctx := autograd.ctx[f64]()
	x := f64_ctx.variable(vtl.from_1d([1.0, 2.0])!)
	for axis in [1, -2] {
		_ := autograd.concatenate[f64]([x], axis: axis) or {
			assert err.msg().contains('axis out of range')
			continue
		}
		assert false, 'expected concatenate to reject axis ${axis}'
	}
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

fn test_autograd_stack_rejects_different_input_shapes() {
	f64_ctx := autograd.ctx[f64]()
	x := f64_ctx.variable(vtl.from_2d([[1.0, 2.0], [3.0, 4.0]])!)
	y := f64_ctx.variable(vtl.from_1d([5.0, 6.0])!)
	_ := autograd.stack[f64]([x, y], axis: 2) or {
		assert err.msg().contains('same shape')
		return
	}
	assert false, 'expected stack to reject different input shapes'
}

fn test_reshape_and_transpose_preserve_nonuniform_gradient_values() ! {
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
