module layers

import vtl
import vtl.autograd
import math

fn activations_ctx[T]() &autograd.Context[T] {
	return autograd.ctx[T]()
}

fn activations_variable[T](c &autograd.Context[T], arr []T, shape []int) !&autograd.Variable[T] {
	t := vtl.from_array(arr, shape)!
	return c.variable(t)
}

fn test_softplus_forward_and_backward() ! {
	c := activations_ctx[f64]()
	layer := softplus_layer[f64](c, [3])
	input := activations_variable[f64](c, [-2.0, 0.0, 2.0], [3])!
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
	c := activations_ctx[f64]()
	layer := selu_layer[f64](c, [3])
	input := activations_variable[f64](c, [-1.0, 0.0, 1.0], [3])!
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
	c := activations_ctx[f64]()
	layer := hardswish_layer[f64](c, [6])
	input := activations_variable[f64](c, [-4.0, -3.0, -1.0, 0.0, 3.0, 4.0], [6])!
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
	gelu_ctx := activations_ctx[f64]()
	gelu_input := activations_variable[f64](gelu_ctx, values, [3])!
	mut gelu_output := gelu_layer[f64](gelu_ctx, [3]).forward(gelu_input)!
	gelu_output.backprop()!
	for i, x in values {
		z := 0.7978845608028654 * (x + 0.044715 * x * x * x)
		tanh_z := math.tanh(z)
		dz_dx := 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x * x)
		expected := 0.5 * (1.0 + tanh_z + x * (1.0 - tanh_z * tanh_z) * dz_dx)
		assert math.abs(gelu_input.grad.get_nth(i) - expected) < 1e-12
	}

	swish_ctx := activations_ctx[f64]()
	swish_input := activations_variable[f64](swish_ctx, values, [3])!
	mut swish_output := swish_layer[f64](swish_ctx, [3]).forward(swish_input)!
	swish_output.backprop()!
	for i, x in values {
		sigmoid := 1.0 / (1.0 + math.exp(-x))
		expected := sigmoid * (1.0 + x * (1.0 - sigmoid))
		assert math.abs(swish_input.grad.get_nth(i) - expected) < 1e-12
	}

	mish_ctx := activations_ctx[f64]()
	mish_input := activations_variable[f64](mish_ctx, values, [3])!
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
	gelu_ctx := activations_ctx[f64]()
	gelu_input := activations_variable[f64](gelu_ctx, values, [2])!
	gelu_output := gelu_layer[f64](gelu_ctx, [2]).forward(gelu_input)!
	assert gelu_output.value.get_nth(0) == 0.0
	assert gelu_output.value.get_nth(1) == 1000.0

	mish_ctx := activations_ctx[f64]()
	mish_input := activations_variable[f64](mish_ctx, values, [2])!
	mish_output := mish_layer[f64](mish_ctx, [2]).forward(mish_input)!
	assert mish_output.value.get_nth(0) == 0.0
	assert mish_output.value.get_nth(1) == 1000.0
}

fn test_activations_handle_large_finite_inputs() ! {
	c := activations_ctx[f64]()
	softplus_input := activations_variable[f64](c, [-1000.0, 1000.0], [2])!
	softplus_output := softplus_layer[f64](c, [2]).forward(softplus_input)!
	assert softplus_output.value.get_nth(0) == 0.0
	assert softplus_output.value.get_nth(1) == 1000.0

	selu_input := activations_variable[f64](c, [-1000.0, 1000.0], [2])!
	selu_output := selu_layer[f64](c, [2]).forward(selu_input)!
	assert math.abs(selu_output.value.get_nth(0) + 1.75809934) < 1e-6
	assert math.abs(selu_output.value.get_nth(1) - 1050.700987) < 1e-6

	hardswish_input := activations_variable[f64](c, [-1000.0, 1000.0, 1e308], [3])!
	hardswish_output := hardswish_layer[f64](c, [2]).forward(hardswish_input)!
	assert hardswish_output.value.get_nth(0) == 0.0
	assert hardswish_output.value.get_nth(1) == 1000.0
	assert hardswish_output.value.get_nth(2) == 1e308
}
