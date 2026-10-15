module layers

import vtl
import vtl.autograd
import math

fn activations_f32_ctx[T]() &autograd.Context[T] {
	return autograd.ctx[T]()
}

fn activations_f32_variable[T](c &autograd.Context[T], arr []T, shape []int) !&autograd.Variable[T] {
	t := vtl.from_array(arr, shape)!
	return c.variable(t)
}

fn test_activations_support_f32_forward_and_backward() ! {
	softplus_ctx := activations_f32_ctx[f32]()
	softplus_input := activations_f32_variable[f32](softplus_ctx, [-1.0, 1.0], [2])!
	mut softplus_output := softplus_layer[f32](softplus_ctx, [2]).forward(softplus_input)!
	assert math.abs(f64(softplus_output.value.get_nth(0)) - 0.313261688) < 1e-6
	assert math.abs(f64(softplus_output.value.get_nth(1)) - 1.313261688) < 1e-6
	softplus_output.backprop()!
	assert math.abs(f64(softplus_input.grad.get_nth(0)) - 0.268941432) < 1e-6
	assert math.abs(f64(softplus_input.grad.get_nth(1)) - 0.731058598) < 1e-6

	selu_ctx := activations_f32_ctx[f32]()
	selu_input := activations_f32_variable[f32](selu_ctx, [-1.0, 1.0], [2])!
	mut selu_output := selu_layer[f32](selu_ctx, [2]).forward(selu_input)!
	selu_output.backprop()!
	assert math.abs(f64(selu_input.grad.get_nth(0)) - 0.6467686) < 1e-6
	assert math.abs(f64(selu_input.grad.get_nth(1)) - 1.0507009) < 1e-6

	hardswish_ctx := activations_f32_ctx[f32]()
	hardswish_input := activations_f32_variable[f32](hardswish_ctx, [-1.0, 1.0], [2])!
	mut hardswish_output := hardswish_layer[f32](hardswish_ctx, [2]).forward(hardswish_input)!
	hardswish_output.backprop()!
	assert math.abs(f64(hardswish_input.grad.get_nth(0)) - 1.0 / 6.0) < 1e-6
	assert math.abs(f64(hardswish_input.grad.get_nth(1)) - 5.0 / 6.0) < 1e-6
}
