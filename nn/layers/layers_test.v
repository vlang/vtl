module layers

import vtl
import vtl.autograd
import math

fn ctx[T]() &autograd.Context[T] {
	return autograd.ctx[T]()
}

fn variable[T](c &autograd.Context[T], arr []T, shape []int) !&autograd.Variable[T] {
	t := vtl.from_array(arr, shape)!
	return c.variable(t)
}

// Linear layer: output shape and forward pass
fn test_linear_output_shape() {
	c := ctx[f64]()
	layer := linear_layer[f64](c, 4, 3)
	assert layer.output_shape() == [3], 'linear output_shape expected [3], got ${layer.output_shape()}'
}

fn test_linear_forward_shape() ! {
	c := ctx[f64]()
	layer := linear_layer[f64](c, 4, 3)
	input := variable[f64](c, [1.0, 0.0, -1.0, 0.5], [1, 4])!
	result := layer.forward(input)!
	// output should be [1, 3]
	assert result.value.shape == [1, 3], 'linear forward shape expected [1, 3], got ${result.value.shape}'
}

fn test_linear_variables_count() {
	c := ctx[f64]()
	layer := linear_layer[f64](c, 4, 3)
	vars := layer.variables()
	assert vars.len == 2, 'linear should have 2 variables (weights + bias), got ${vars.len}'
}

// BatchNorm layer: output shape
fn test_batchnorm_output_shape() {
	c := ctx[f64]()
	layer := batchnorm1d_layer[f64](c, 8, BatchNorm1DConfig{})
	assert layer.output_shape() == [8], 'batchnorm output_shape expected [8], got ${layer.output_shape()}'
}

fn test_batchnorm_forward_shape() ! {
	c := ctx[f64]()
	layer := batchnorm1d_layer[f64](c, 4, BatchNorm1DConfig{})
	input := variable[f64](c, [1.0, 2.0, 3.0, 4.0], [1, 4])!
	result := layer.forward(input)!
	assert result.value.shape == [1, 4], 'batchnorm forward shape expected [1, 4], got ${result.value.shape}'
}

fn test_batchnorm_variables_count() {
	c := ctx[f64]()
	layer := batchnorm1d_layer[f64](c, 4, BatchNorm1DConfig{})
	vars := layer.variables()
	assert vars.len == 2, 'batchnorm should have 2 variables (gamma + beta), got ${vars.len}'
}

fn test_batchnorm_training_backward_includes_variance_gradient() ! {
	c := ctx[f64]()
	layer := batchnorm1d_layer[f64](c, 1, BatchNorm1DConfig{})
	input := variable[f64](c, [1.0, 3.0], [2, 1])!
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
	c := ctx[f64]()
	layer := batchnorm1d_layer[f64](c, 1, BatchNorm1DConfig{})
	input := variable[f64](c, [1.0, 3.0], [2, 1])!
	output := layer.forward_with_mode(input, false)!
	weights := c.variable(vtl.from_array([1.0, 0.0], [2, 1])!, requires_grad: false)
	mut loss := output.multiply(weights)!.sum()!
	loss.backprop()!

	assert math.abs(input.grad.get_nth(0) - 1.0 / math.sqrt(1.0 + 1e-5)) < 1e-12
	assert input.grad.get_nth(1) == 0.0
}

fn test_avgpool_backward_propagates_input_gradients() ! {
	c := ctx[f64]()
	input := variable[f64](c, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [1, 1, 3, 3])!
	layer := avgpool2d_layer[f64](c, [1, 3, 3], [2, 2], [0, 0], [1, 1])
	mut output := layer.forward(input)!
	output.backprop()!

	assert input.grad.to_array() == [0.25, 0.5, 0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.25]
}

// Embedding layer: output shape and variables
fn test_embedding_output_shape() {
	c := ctx[f64]()
	layer := embedding_layer[f64](c, 100, 16)
	assert layer.output_shape() == [16], 'embedding output_shape expected [16], got ${layer.output_shape()}'
}

fn test_embedding_variables_count() {
	c := ctx[f64]()
	layer := embedding_layer[f64](c, 100, 16)
	vars := layer.variables()
	assert vars.len == 1, 'embedding should have 1 variable (weight), got ${vars.len}'
	assert vars[0].value.shape == [100, 16], 'embedding weight shape expected [100, 16], got ${vars[0].value.shape}'
}

fn test_softplus_forward_and_backward() ! {
	c := ctx[f64]()
	layer := softplus_layer[f64](c, [3])
	input := variable[f64](c, [-2.0, 0.0, 2.0], [3])!
	mut output := layer.forward(input)!
	assert math.abs(output.value.get_nth(0) - 0.126928011) < 1e-6
	assert math.abs(output.value.get_nth(1) - 0.693147181) < 1e-6
	assert math.abs(output.value.get_nth(2) - 2.126928011) < 1e-6
	output.backprop()!
	assert math.abs(input.grad.get_nth(0) - 0.119202922) < 1e-6
	assert math.abs(input.grad.get_nth(1) - 0.5) < 1e-6
	assert math.abs(input.grad.get_nth(2) - 0.880797078) < 1e-6
}

fn test_selu_forward_and_backward() ! {
	c := ctx[f64]()
	layer := selu_layer[f64](c, [3])
	input := variable[f64](c, [-1.0, 0.0, 1.0], [3])!
	mut output := layer.forward(input)!
	assert math.abs(output.value.get_nth(0) - (-1.111330737)) < 1e-6
	assert output.value.get_nth(1) == 0.0
	assert math.abs(output.value.get_nth(2) - 1.050700987) < 1e-6
	output.backprop()!
	assert math.abs(input.grad.get_nth(0) - 0.646768605) < 1e-6
	assert math.abs(input.grad.get_nth(1) - 1.758099341) < 1e-6
	assert math.abs(input.grad.get_nth(2) - 1.050700987) < 1e-6
}

fn test_hardswish_forward_and_backward() ! {
	c := ctx[f64]()
	layer := hardswish_layer[f64](c, [6])
	input := variable[f64](c, [-4.0, -3.0, -1.0, 0.0, 3.0, 4.0], [6])!
	mut output := layer.forward(input)!
	assert output.value.get_nth(0) == 0.0
	assert output.value.get_nth(1) == 0.0
	assert math.abs(output.value.get_nth(2) + 1.0 / 3.0) < 1e-12
	assert output.value.get_nth(3) == 0.0
	assert output.value.get_nth(4) == 3.0
	assert output.value.get_nth(5) == 4.0
	output.backprop()!
	assert input.grad.get_nth(0) == 0.0
	assert input.grad.get_nth(1) == 0.0
	assert math.abs(input.grad.get_nth(2) - 1.0 / 6.0) < 1e-12
	assert input.grad.get_nth(3) == 0.5
	assert input.grad.get_nth(4) == 1.0
	assert input.grad.get_nth(5) == 1.0
}

fn test_gelu_swish_and_mish_backward_use_input_values() ! {
	values := [-1.0, 0.0, 1.0]
	gelu_ctx := ctx[f64]()
	gelu_input := variable[f64](gelu_ctx, values, [3])!
	mut gelu_output := gelu_layer[f64](gelu_ctx, [3]).forward(gelu_input)!
	gelu_output.backprop()!
	for i, x in values {
		z := 0.7978845608028654 * (x + 0.044715 * x * x * x)
		tanh_z := math.tanh(z)
		dz_dx := 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x * x)
		expected := 0.5 * (1.0 + tanh_z + x * (1.0 - tanh_z * tanh_z) * dz_dx)
		assert math.abs(gelu_input.grad.get_nth(i) - expected) < 1e-12
	}

	swish_ctx := ctx[f64]()
	swish_input := variable[f64](swish_ctx, values, [3])!
	mut swish_output := swish_layer[f64](swish_ctx, [3]).forward(swish_input)!
	swish_output.backprop()!
	for i, x in values {
		sigmoid := 1.0 / (1.0 + math.exp(-x))
		expected := sigmoid * (1.0 + x * (1.0 - sigmoid))
		assert math.abs(swish_input.grad.get_nth(i) - expected) < 1e-12
	}

	mish_ctx := ctx[f64]()
	mish_input := variable[f64](mish_ctx, values, [3])!
	mut mish_output := mish_layer[f64](mish_ctx, [3]).forward(mish_input)!
	mish_output.backprop()!
	for i, x in values {
		softplus := math.log1p(math.exp(x))
		tanh_softplus := math.tanh(softplus)
		sigmoid := 1.0 / (1.0 + math.exp(-x))
		expected := tanh_softplus + x * (1.0 - tanh_softplus * tanh_softplus) * sigmoid
		assert math.abs(mish_input.grad.get_nth(i) - expected) < 1e-12
	}
}

fn test_gelu_and_mish_are_finite_for_large_inputs() ! {
	values := [-1000.0, 1000.0]
	gelu_ctx := ctx[f64]()
	gelu_input := variable[f64](gelu_ctx, values, [2])!
	gelu_output := gelu_layer[f64](gelu_ctx, [2]).forward(gelu_input)!
	assert gelu_output.value.get_nth(0) == 0.0
	assert gelu_output.value.get_nth(1) == 1000.0

	mish_ctx := ctx[f64]()
	mish_input := variable[f64](mish_ctx, values, [2])!
	mish_output := mish_layer[f64](mish_ctx, [2]).forward(mish_input)!
	assert mish_output.value.get_nth(0) == 0.0
	assert mish_output.value.get_nth(1) == 1000.0
}

fn test_activations_support_f32_forward_and_backward() ! {
	softplus_ctx := ctx[f32]()
	softplus_input := variable[f32](softplus_ctx, [-1.0, 1.0], [2])!
	mut softplus_output := softplus_layer[f32](softplus_ctx, [2]).forward(softplus_input)!
	assert math.abs(f64(softplus_output.value.get_nth(0)) - 0.313261688) < 1e-6
	assert math.abs(f64(softplus_output.value.get_nth(1)) - 1.313261688) < 1e-6
	softplus_output.backprop()!
	assert math.abs(f64(softplus_input.grad.get_nth(0)) - 0.268941432) < 1e-6
	assert math.abs(f64(softplus_input.grad.get_nth(1)) - 0.731058598) < 1e-6

	selu_ctx := ctx[f32]()
	selu_input := variable[f32](selu_ctx, [-1.0, 1.0], [2])!
	mut selu_output := selu_layer[f32](selu_ctx, [2]).forward(selu_input)!
	selu_output.backprop()!
	assert math.abs(f64(selu_input.grad.get_nth(0)) - 0.6467686) < 1e-6
	assert math.abs(f64(selu_input.grad.get_nth(1)) - 1.0507009) < 1e-6

	hardswish_ctx := ctx[f32]()
	hardswish_input := variable[f32](hardswish_ctx, [-1.0, 1.0], [2])!
	mut hardswish_output := hardswish_layer[f32](hardswish_ctx, [2]).forward(hardswish_input)!
	hardswish_output.backprop()!
	assert math.abs(f64(hardswish_input.grad.get_nth(0)) - 1.0 / 6.0) < 1e-6
	assert math.abs(f64(hardswish_input.grad.get_nth(1)) - 5.0 / 6.0) < 1e-6
}

fn test_activations_handle_large_finite_inputs() ! {
	c := ctx[f64]()
	softplus_input := variable[f64](c, [-1000.0, 1000.0], [2])!
	softplus_output := softplus_layer[f64](c, [2]).forward(softplus_input)!
	assert softplus_output.value.get_nth(0) == 0.0
	assert softplus_output.value.get_nth(1) == 1000.0

	selu_input := variable[f64](c, [-1000.0, 1000.0], [2])!
	selu_output := selu_layer[f64](c, [2]).forward(selu_input)!
	assert math.abs(selu_output.value.get_nth(0) + 1.75809934) < 1e-6
	assert math.abs(selu_output.value.get_nth(1) - 1050.700987) < 1e-6

	hardswish_input := variable[f64](c, [-1000.0, 1000.0, 1e308], [3])!
	hardswish_output := hardswish_layer[f64](c, [2]).forward(hardswish_input)!
	assert hardswish_output.value.get_nth(0) == 0.0
	assert hardswish_output.value.get_nth(1) == 1000.0
	assert hardswish_output.value.get_nth(2) == 1e308
}
