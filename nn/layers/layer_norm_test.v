module layers

import vtl
import vtl.autograd
import math

fn test_layer_norm_layer_uses_normalized_shape_for_forward_and_backward() ! {
	ctx := autograd.ctx[f64]()
	layer := layer_norm_layer[f64](ctx, [3], LayerNormConfig{})
	input := ctx.variable(vtl.from_array([1.0, 2.0, 4.0, 0.0, 3.0, 8.0], [2, 3])!)
	mut output := layer.forward(input)!
	assert output.value.shape == [2, 3]
	assert math.abs(output.value.get_nth(0) - ((1.0 - 7.0 / 3.0) / math.sqrt(14.0 / 9.0 + 1e-5))) < 1e-12

	loss := output.sum()!
	loss.backprop()!
	assert input.grad.shape == [2, 3]
	parameters := layer.variables()
	assert parameters.len == 2
	assert parameters[0].grad.shape == [3]
	assert parameters[1].grad.to_array() == [2.0, 2.0, 2.0]
}

fn test_layer_norm_layer_without_affine_parameters() ! {
	ctx := autograd.ctx[f64]()
	layer := layer_norm_layer[f64](ctx, [2], LayerNormConfig{
		affine: false
	})
	input := ctx.variable(vtl.from_array([1.0, 3.0, 4.0, 8.0], [2, 2])!)
	output := layer.forward(input)!
	assert output.value.shape == [2, 2]
	assert math.abs(output.value.get_nth(0) + 0.999995) < 1e-5
	assert math.abs(output.value.get_nth(2) + 0.99999875) < 1e-5
}
