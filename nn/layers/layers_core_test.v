module layers

import vtl
import vtl.autograd
import math

fn core_ctx[T]() &autograd.Context[T] {
	return autograd.ctx[T]()
}

fn core_variable[T](c &autograd.Context[T], arr []T, shape []int) !&autograd.Variable[T] {
	t := vtl.from_array(arr, shape)!
	return c.variable(t)
}

fn test_linear_output_shape() {
	c := core_ctx[f64]()
	layer := linear_layer[f64](c, 4, 3)
	assert layer.output_shape() == [3], 'linear output_shape expected [3], got ${layer.output_shape()}'
}

fn test_linear_forward_shape() ! {
	c := core_ctx[f64]()
	layer := linear_layer[f64](c, 4, 3)
	input := core_variable[f64](c, [1.0, 0.0, -1.0, 0.5], [1, 4])!
	result := layer.forward(input)!
	// output should be [1, 3]
	assert result.value.shape == [1, 3], 'linear forward shape expected [1, 3], got ${result.value.shape}'
}

fn test_linear_variables_count() {
	c := core_ctx[f64]()
	layer := linear_layer[f64](c, 4, 3)
	vars := layer.variables()
	assert vars.len == 2, 'linear should have 2 variables (weights + bias), got ${vars.len}'
}

// BatchNorm layer: output shape

fn test_batchnorm_output_shape() {
	c := core_ctx[f64]()
	layer := batchnorm1d_layer[f64](c, 8, BatchNorm1DConfig{})
	assert layer.output_shape() == [8], 'batchnorm output_shape expected [8], got ${layer.output_shape()}'
}

fn test_batchnorm_forward_shape() ! {
	c := core_ctx[f64]()
	layer := batchnorm1d_layer[f64](c, 4, BatchNorm1DConfig{})
	input := core_variable[f64](c, [1.0, 2.0, 3.0, 4.0], [1, 4])!
	result := layer.forward(input)!
	assert result.value.shape == [1, 4], 'batchnorm forward shape expected [1, 4], got ${result.value.shape}'
}

fn test_batchnorm_variables_count() {
	c := core_ctx[f64]()
	layer := batchnorm1d_layer[f64](c, 4, BatchNorm1DConfig{})
	vars := layer.variables()
	assert vars.len == 2, 'batchnorm should have 2 variables (gamma + beta), got ${vars.len}'
}

fn test_batchnorm_training_backward_includes_variance_gradient() ! {
	c := core_ctx[f64]()
	layer := batchnorm1d_layer[f64](c, 1, BatchNorm1DConfig{})
	input := core_variable[f64](c, [1.0, 3.0], [2, 1])!
	output := layer.forward_with_mode(input, true)!
	weights := c.variable(vtl.from_array([1.0, 0.0], [2, 1])!, requires_grad: false)
	mut loss := output.multiply(weights)!.sum()!
	loss.backprop()!

	// The centered two-element batch has nearly zero input gradient when one
	// output element receives a unit gradient. Omitting d(variance)/dx yields
	// gradients near +0.5 and -0.5 instead.
	assert math.abs(input.grad.get_nth(0)) < 1e-4
	assert math.abs(input.grad.get_nth(1)) < 1e-4
}

fn test_batchnorm_eval_backward_treats_running_stats_as_constant() ! {
	c := core_ctx[f64]()
	layer := batchnorm1d_layer[f64](c, 1, BatchNorm1DConfig{})
	input := core_variable[f64](c, [1.0, 3.0], [2, 1])!
	output := layer.forward_with_mode(input, false)!
	weights := c.variable(vtl.from_array([1.0, 0.0], [2, 1])!, requires_grad: false)
	mut loss := output.multiply(weights)!.sum()!
	loss.backprop()!

	assert math.abs(input.grad.get_nth(0) - 1.0 / math.sqrt(1.0 + 1e-5)) < 1e-12
	assert input.grad.get_nth(1) == 0.0
}

fn test_avgpool_backward_propagates_input_gradients() ! {
	c := core_ctx[f64]()
	input := core_variable[f64](c, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [1, 1, 3, 3])!
	layer := avgpool2d_layer[f64](c, [1, 3, 3], [2, 2], [0, 0], [1, 1])
	mut output := layer.forward(input)!
	output.backprop()!

	assert input.grad.to_array() == [0.25, 0.5, 0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.25]
}

// Embedding layer: output shape and variables

fn test_embedding_output_shape() {
	c := core_ctx[f64]()
	layer := embedding_layer[f64](c, 100, 16)
	assert layer.output_shape() == [16], 'embedding output_shape expected [16], got ${layer.output_shape()}'
}

fn test_embedding_variables_count() {
	c := core_ctx[f64]()
	layer := embedding_layer[f64](c, 100, 16)
	vars := layer.variables()
	assert vars.len == 1, 'embedding should have 1 variable (weight), got ${vars.len}'
	assert vars[0].value.shape == [100, 16], 'embedding weight shape expected [100, 16], got ${vars[0].value.shape}'
}
