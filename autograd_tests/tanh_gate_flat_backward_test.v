module autograd_tests

import vtl.autograd
import vtl

fn test_tanh_gate_backward_contiguous_flat_path() ! {
	ctx := autograd.ctx[f64]()
	gradient := vtl.from_1d([2.0, 4.0, 6.0])!
	cached := vtl.from_1d([-0.5, 0.0, 0.5])!
	mut output := ctx.variable(vtl.zeros[f64]([3]))
	output.grad = gradient
	result := autograd.tanh_gate[f64](cached).backward(autograd.payload(output))!

	assert result[0].array_equal(vtl.from_1d([1.5, 4.0, 4.5])!)
}

fn test_tanh_gate_backward_contiguous_f32() ! {
	ctx := autograd.ctx[f32]()
	gradient := vtl.from_1d[f32]([2.0, 4.0])!
	cached := vtl.from_1d[f32]([-0.5, 0.5])!
	mut output := ctx.variable(vtl.zeros[f32]([2]))
	output.grad = gradient
	result := autograd.tanh_gate[f32](cached).backward(autograd.payload(output))!

	assert result[0].array_equal(vtl.from_1d[f32]([1.5, 3.0])!)
}

fn test_tanh_gate_backward_non_contiguous_fallback() ! {
	ctx := autograd.ctx[f64]()
	gradient := vtl.from_array[f64]([2.0, 4.0, 6.0, 8.0], [2, 2])!.transpose([1, 0])!
	cached := vtl.from_array[f64]([-0.5, 0.0, 0.5, 1.0], [2, 2])!.transpose([1, 0])!
	mut output := ctx.variable(vtl.zeros[f64]([2, 2]))
	output.grad = gradient
	result := autograd.tanh_gate[f64](cached).backward(autograd.payload(output))!

	assert result[0].get([0, 0]) == 1.5
	assert result[0].get([0, 1]) == 4.5
	assert result[0].get([1, 0]) == 4.0
	assert result[0].get([1, 1]) == 0.0
}

fn test_tanh_gate_backward_broadcast_fallback() ! {
	ctx := autograd.ctx[f64]()
	gradient := vtl.from_array[f64]([2.0, 4.0, 6.0, 8.0], [2, 2])!
	cached := vtl.from_1d([0.0, 0.5])!
	mut output := ctx.variable(vtl.zeros[f64]([2, 2]))
	output.grad = gradient
	result := autograd.tanh_gate[f64](cached).backward(autograd.payload(output))!

	assert result[0].array_equal(vtl.from_array[f64]([2.0, 3.0, 6.0, 6.0], [2, 2])!)
}
