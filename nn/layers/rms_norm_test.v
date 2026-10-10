module layers

import math
import vtl
import vtl.autograd

fn test_rms_norm_layer_forward_backward_and_weight() ! {
	ctx := autograd.ctx[f64]()
	layer := rms_norm_layer[f64](ctx, [2], RMSNormConfig{
		eps: 1e-5
	})
	input := ctx.variable(vtl.from_array([3.0, 4.0, 5.0, 12.0], [2, 2])!)
	output := layer.forward(input)!
	assert output.value.shape == [2, 2]
	assert math.abs(output.value.get_nth(0) - 3.0 / math.sqrt(12.5 + 1e-5)) < 1e-6
	mut loss := output.sum()!
	loss.backprop()!
	assert input.grad.shape == [2, 2]
	assert layer.variables().len == 1
	assert layer.variables()[0].grad.shape == [2]
}

fn test_rms_norm_layer_without_affine_parameters() ! {
	ctx := autograd.ctx[f32]()
	layer := rms_norm_layer[f32](ctx, [2], RMSNormConfig{
		elementwise_affine: false
	})
	input := ctx.variable(vtl.from_array([f32(3), 4], [1, 2])!)
	output := layer.forward(input)!
	assert layer.variables().len == 0
	expected := 3.0 / math.sqrt(12.5 + 1.1920928955078125e-7)
	assert math.abs(f64(output.value.get_nth(0)) - expected) < 1e-7
	mut loss := output.sum()!
	loss.backprop()!
	assert input.grad.shape == [1, 2]
}
