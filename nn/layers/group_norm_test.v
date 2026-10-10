module layers

import vtl
import vtl.autograd
import math

fn test_group_norm_layer_forward_backward_and_affine_parameters() ! {
	ctx := autograd.ctx[f64]()
	layer := group_norm_layer[f64](ctx, [2, 2], 1, GroupNormConfig{})
	input := ctx.variable(vtl.from_array([1.0, 2.0, 4.0, 0.0], [1, 2, 2])!)
	output := layer.forward(input)!
	assert output.value.shape == [1, 2, 2]
	assert math.abs(output.value.get_nth(0) + 0.507091) < 1e-5
	loss := output.sum()!
	loss.backprop()!
	assert input.grad.shape == [1, 2, 2]
	parameters := layer.variables()
	assert parameters.len == 2
	assert parameters[0].grad.shape == [2]
	assert parameters[1].grad.to_array() == [2.0, 2.0]
}

fn test_group_norm_layer_without_affine_parameters_and_shape_validation() ! {
	ctx := autograd.ctx[f32]()
	layer := group_norm_layer[f32](ctx, [2, 2], 2, GroupNormConfig{
		affine: false
	})
	input := ctx.variable(vtl.from_array([f32(1), 3, 4, 8], [1, 2, 2])!)
	output := layer.forward(input)!
	assert layer.variables().len == 0
	assert output.value.get_nth(0) < -0.99
	wrong_shape := ctx.variable(vtl.from_array([f32(1), 2, 3], [1, 3])!)
	if _ := layer.forward(wrong_shape) {
		assert false, 'expected mismatched input shape to fail'
	}
}
