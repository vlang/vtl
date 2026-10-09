module layers

import vtl
import vtl.autograd

fn test_linear_bias_gradient_sums_the_f64_batch() ! {
	ctx := autograd.ctx[f64]()
	layer := linear_layer[f64](ctx, 2, 2)
	input := ctx.variable(vtl.from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2])!)
	mut loss := layer.forward(input)!.sum()!
	loss.backprop()!
	bias_gradient := layer.variables()[1].grad
	assert bias_gradient.shape == [1, 2]
	assert bias_gradient.to_array() == [3.0, 3.0]
}

fn test_linear_bias_gradient_sums_the_f32_batch() ! {
	ctx := autograd.ctx[f32]()
	layer := linear_layer[f32](ctx, 2, 2)
	input := ctx.variable(vtl.from_array([f32(1), 2, 3, 4, 5, 6], [3, 2])!)
	mut loss := layer.forward(input)!.sum()!
	loss.backprop()!
	bias_gradient := layer.variables()[1].grad
	assert bias_gradient.shape == [1, 2]
	assert bias_gradient.to_array() == [3.0, 3.0]
}
