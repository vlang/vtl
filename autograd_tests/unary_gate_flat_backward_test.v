module autograd_tests

import vtl.autograd
import vtl
import math

fn test_tanh_gate_backward_contiguous_flat_path() ! {
	ctx := autograd.ctx[f64]()
	gradient := vtl.from_1d([2.0, 4.0, 6.0, 8.0, 10.0])!
	cached := vtl.from_1d([-0.5, 0.0, 0.5, 1.0, -1.0])!
	mut output := ctx.variable(vtl.zeros[f64]([5]))
	output.grad = gradient
	result := autograd.tanh_gate[f64](cached).backward(autograd.payload(output))!

	assert result[0].array_equal(vtl.from_1d([1.5, 4.0, 4.5, 0.0, 0.0])!)
}

fn test_tanh_gate_backward_contiguous_f32() ! {
	ctx := autograd.ctx[f32]()
	gradient := vtl.from_1d[f32]([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0])!
	cached := vtl.from_1d[f32]([-0.5, 0.0, 0.5, 1.0, -1.0, 0.25, -0.25, 0.75, -0.75])!
	mut output := ctx.variable(vtl.zeros[f32]([9]))
	output.grad = gradient
	result := autograd.tanh_gate[f32](cached).backward(autograd.payload(output))!

	assert result[0].array_equal(vtl.from_1d[f32]([1.5, 4.0, 4.5, 0.0, 0.0, 11.25, 13.125, 7.0,
		7.875])!)
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

fn test_abs_gate_backward_contiguous_flat_path() ! {
	ctx := autograd.ctx[f64]()
	input := ctx.variable(vtl.from_1d([-2.0, 0.0, 1e-9, 2.0])!)
	gradient := vtl.from_1d([2.0, 4.0, 6.0, 8.0])!
	mut output := ctx.variable(vtl.zeros[f64]([4]))
	output.grad = gradient
	result := autograd.abs_gate[f64](input).backward(autograd.payload(output))!

	assert result[0].get_nth(0) == -2.0
	assert result[0].get_nth(1) == 0.0
	assert math.abs(result[0].get_nth(2) - 0.6) < 1e-12
	assert result[0].get_nth(3) == 8.0
}

fn test_abs_gate_backward_non_contiguous_fallback() ! {
	ctx := autograd.ctx[f64]()
	input := ctx.variable(vtl.from_array([-2.0, 0.0, 1e-9, 2.0], [2, 2])!.transpose([
		1,
		0,
	])!)
	gradient := vtl.from_array([2.0, 4.0, 6.0, 8.0], [2, 2])!.transpose([1, 0])!
	mut output := ctx.variable(vtl.zeros[f64]([2, 2]))
	output.grad = gradient
	result := autograd.abs_gate[f64](input).backward(autograd.payload(output))!

	assert result[0].get([0, 0]) == -2.0
	assert math.abs(result[0].get([0, 1]) - 0.6) < 1e-12
	assert result[0].get([1, 0]) == 0.0
	assert result[0].get([1, 1]) == 8.0
}

fn test_sqrt_gate_backward_contiguous_flat_path() ! {
	ctx := autograd.ctx[f64]()
	input := ctx.variable(vtl.from_1d([1.0, 4.0, 9.0, 16.0, 25.0])!)
	gradient := vtl.from_1d([2.0, 4.0, 6.0, 8.0, 10.0])!
	mut output := ctx.variable(vtl.zeros[f64]([5]))
	output.grad = gradient
	result := autograd.sqrt_gate[f64](input).backward(autograd.payload(output))!

	assert result[0].array_equal(vtl.from_1d([1.0, 1.0, 1.0, 1.0, 1.0])!)
}

fn test_sqrt_gate_backward_broadcast_fallback() ! {
	ctx := autograd.ctx[f64]()
	input := ctx.variable(vtl.from_1d([1.0, 4.0])!)
	gradient := vtl.from_array([2.0, 4.0, 6.0, 8.0], [2, 2])!
	mut output := ctx.variable(vtl.zeros[f64]([2, 2]))
	output.grad = gradient
	result := autograd.sqrt_gate[f64](input).backward(autograd.payload(output))!

	assert result[0].get([0, 0]) == 1.0
	assert result[0].get([0, 1]) == 1.0
	assert result[0].get([1, 0]) == 3.0
	assert result[0].get([1, 1]) == 2.0
}

fn test_clamp_gate_backward_contiguous_flat_path() ! {
	ctx := autograd.ctx[f64]()
	input := vtl.from_1d([-2.0, -1.0, 0.0, 1.0, 2.0])!
	gradient := vtl.from_1d([1.0, 2.0, 3.0, 4.0, 5.0])!
	mut output := ctx.variable(vtl.zeros[f64]([5]))
	output.grad = gradient
	result := autograd.clamp_gate[f64](-1.0, 1.0, input).backward(autograd.payload(output))!

	assert result[0].array_equal(vtl.from_1d([0.0, 2.0, 3.0, 4.0, 0.0])!)
}

fn test_clamp_gate_backward_non_contiguous_fallback() ! {
	input := vtl.from_array([-2.0, -1.0, 0.0, 2.0], [2, 2])!.transpose([1, 0])!
	gradient := vtl.from_array([1.0, 2.0, 3.0, 4.0], [2, 2])!.transpose([1, 0])!
	ctx := autograd.ctx[f64]()
	mut output := ctx.variable(vtl.zeros[f64]([2, 2]))
	output.grad = gradient
	result := autograd.clamp_gate[f64](-1.0, 1.0, input).backward(autograd.payload(output))!

	assert result[0].get([0, 0]) == 0.0
	assert result[0].get([0, 1]) == 3.0
	assert result[0].get([1, 0]) == 2.0
	assert result[0].get([1, 1]) == 0.0
}

fn test_f32_unary_gate_simd_blocks_and_tails() ! {
	ctx := autograd.ctx[f32]()
	mut output := ctx.variable(vtl.zeros[f32]([9]))
	output.grad = vtl.from_1d[f32]([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0])!

	tanh_cache := vtl.from_1d[f32]([-0.5, 0.0, 0.5, 1.0, -1.0, 0.25, -0.25, 0.75, -0.75])!
	tanh_result := autograd.tanh_gate[f32](tanh_cache).backward(autograd.payload(output))!
	assert tanh_result[0].array_equal(vtl.from_1d[f32]([1.5, 4.0, 4.5, 0.0, 0.0, 11.25, 13.125,
		7.0, 7.875])!)

	abs_input := ctx.variable(vtl.from_1d[f32]([-2.0, 0.0, 1e-9, 2.0, -1e-8, 1e-8, -4.0, 4.0, 0.5])!)
	abs_result := autograd.abs_gate[f32](abs_input).backward(autograd.payload(output))!
	assert abs_result[0].get_nth(0) == -2.0
	assert abs_result[0].get_nth(1) == 0.0
	assert math.abs(abs_result[0].get_nth(2) - 0.6) < 1e-5
	assert abs_result[0].get_nth(8) == 18.0

	sqrt_input := ctx.variable(vtl.from_1d[f32]([1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0])!)
	output.grad = vtl.from_1d[f32]([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0])!
	sqrt_result := autograd.sqrt_gate[f32](sqrt_input).backward(autograd.payload(output))!
	assert sqrt_result[0].array_equal(vtl.from_1d[f32]([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
		1.0])!)

	clamp_input := vtl.from_1d[f32]([-2.0, -1.0, f32(math.nan()), 1.0, 2.0, -1.5, 0.5, 1.5, -0.5])!
	output.grad = vtl.from_1d[f32]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])!
	clamp_result := autograd.clamp_gate[f32](-1.0, 1.0, clamp_input).backward(autograd.payload(output))!
	assert clamp_result[0].array_equal(vtl.from_1d[f32]([0.0, 2.0, 0.0, 4.0, 0.0, 0.0, 7.0, 0.0,
		9.0])!)
}

fn test_exp_gate_backward_contiguous_flat_path() ! {
	ctx := autograd.ctx[f64]()
	input := ctx.variable(vtl.from_1d([0.0, 1.0, -1.0, 2.0])!)
	mut output := ctx.variable(vtl.zeros[f64]([4]))
	output.grad = vtl.from_1d([1.0, 2.0, 3.0, 4.0])!
	result := autograd.exp_gate[f64](input).backward(autograd.payload(output))!

	assert result[0].array_equal(vtl.from_1d([1.0, 2.0 * math.exp(1.0), 3.0 * math.exp(-1.0),
		4.0 * math.exp(2.0)])!)
}

fn test_log_gate_backward_contiguous_flat_path() ! {
	ctx := autograd.ctx[f64]()
	input := ctx.variable(vtl.from_1d([1.0, 2.0, 4.0, 0.5])!)
	mut output := ctx.variable(vtl.zeros[f64]([4]))
	output.grad = vtl.from_1d([2.0, 2.0, 8.0, 3.0])!
	result := autograd.log_gate[f64](input).backward(autograd.payload(output))!

	assert result[0].array_equal(vtl.from_1d([2.0, 1.0, 2.0, 6.0])!)
}

fn test_trig_gates_backward_contiguous_flat_path() ! {
	ctx := autograd.ctx[f64]()
	input := ctx.variable(vtl.from_1d([0.0, 1.0, -0.5])!)
	mut output := ctx.variable(vtl.zeros[f64]([3]))
	output.grad = vtl.from_1d([1.0, 2.0, 3.0])!

	sin_result := autograd.sin_gate[f64](input).backward(autograd.payload(output))!
	assert sin_result[0].array_equal(vtl.from_1d([1.0, 2.0 * math.cos(1.0), 3.0 * math.cos(-0.5)])!)

	cos_result := autograd.cos_gate[f64](input).backward(autograd.payload(output))!
	assert cos_result[0].array_equal(vtl.from_1d([0.0, -2.0 * math.sin(1.0), -3.0 * math.sin(-0.5)])!)

	tan_result := autograd.tan_gate[f64](input).backward(autograd.payload(output))!
	assert tan_result[0].array_equal(vtl.from_1d([1.0, 2.0 / (math.cos(1.0) * math.cos(1.0)),
		3.0 / (math.cos(-0.5) * math.cos(-0.5))])!)
}

fn test_exp_gate_backward_non_contiguous_fallback() ! {
	ctx := autograd.ctx[f64]()
	input := ctx.variable(vtl.from_array([0.0, 1.0, -1.0, 2.0], [2, 2])!.transpose([1, 0])!)
	mut output := ctx.variable(vtl.zeros[f64]([2, 2]))
	output.grad = vtl.from_array([1.0, 2.0, 3.0, 4.0], [2, 2])!.transpose([1, 0])!
	result := autograd.exp_gate[f64](input).backward(autograd.payload(output))!

	// input transposed is [[0, -1], [1, 2]]; gradient transposed is [[1, 3], [2, 4]]
	assert result[0].get([0, 0]) == 1.0
	assert result[0].get([0, 1]) == 3.0 * math.exp(-1.0)
	assert result[0].get([1, 0]) == 2.0 * math.exp(1.0)
	assert result[0].get([1, 1]) == 4.0 * math.exp(2.0)
}
