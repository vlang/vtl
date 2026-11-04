module layers

import math
import os
import vtl
import vtl.autograd

// maxpool2d_layer_input_grad runs MaxPool2D forward and backward through autograd and
// returns the input gradient. The caller sets VTL_USE_CUDA and VTL_CUDA_BACKWARD as needed.
fn maxpool2d_layer_input_grad(shape []int, kernel []int, stride []int) ![]f64 {
	mut values := []f64{len: shape[0] * shape[1] * shape[2] * shape[3]}
	for i in 0 .. values.len {
		values[i] = f64((i * 7919) % 1000) / 10.0 - 50.0
	}
	c := autograd.ctx[f64]()
	layer := maxpool2d_layer[f64](c, shape[1..], kernel, [0, 0], stride)
	input := c.variable(vtl.from_array(values, shape)!)
	output := layer.forward(input)!

	mut weight_values := []f64{len: output.value.size}
	for i in 0 .. weight_values.len {
		weight_values[i] = f64(i % 5) - 2.0
	}
	weights := c.variable(vtl.from_array(weight_values, output.value.shape)!, requires_grad: false)
	mut loss := output.multiply(weights)!.sum()!
	loss.backprop()!
	return input.grad.to_array()
}

// Layer-level check: the MaxPool2D gate backward gives the same input gradient on the
// cuDNN path (when enabled) and on the CPU path. Without -d cuda both runs use CPU.
fn test_maxpool2d_layer_backward_cuda_matches_cpu_path() ! {
	os.unsetenv('VTL_USE_CUDA')
	os.unsetenv('VTL_CUDA_BACKWARD')
	cpu_grads := [
		maxpool2d_layer_input_grad([2, 3, 6, 6], [2, 2], [2, 2])!,
		maxpool2d_layer_input_grad([2, 3, 7, 7], [3, 3], [2, 2])!,
	]

	os.setenv('VTL_USE_CUDA', '1', true)
	os.setenv('VTL_CUDA_BACKWARD', '1', true)
	defer {
		os.unsetenv('VTL_USE_CUDA')
		os.unsetenv('VTL_CUDA_BACKWARD')
	}
	gpu_grads := [
		maxpool2d_layer_input_grad([2, 3, 6, 6], [2, 2], [2, 2])!,
		maxpool2d_layer_input_grad([2, 3, 7, 7], [3, 3], [2, 2])!,
	]

	for k in 0 .. cpu_grads.len {
		assert gpu_grads[k].len == cpu_grads[k].len
		for i in 0 .. cpu_grads[k].len {
			assert math.abs(gpu_grads[k][i] - cpu_grads[k][i]) < 1e-9, 'config ${k} grad ${i}: gpu ${gpu_grads[k][i]}, cpu ${cpu_grads[k][i]}'
		}
	}
}
