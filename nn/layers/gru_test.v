module layers

import vtl
import vtl.autograd

fn test_gru_layer_forward_and_autograd() ! {
	ctx := autograd.ctx[f64]()
	layer := gru_layer[f64](ctx, 2, 3)
	assert layer.output_shape() == [3]
	assert layer.variables().len == 4
	input := ctx.variable(vtl.from_array([0.2, -0.1, 0.4, 0.3], [2, 1, 2])!)
	mut output := layer.forward(input)!
	assert output.value.shape == [2, 1, 3]
	output.backprop()!
	assert input.grad.shape == [2, 1, 2]
	mut input_grad_norm := f64(0)
	for value in input.grad.to_array() { input_grad_norm += value * value }
	assert input_grad_norm > 0, 'GRU backward must propagate gradients to its input'
	for parameter in layer.variables() {
		mut grad_norm := f64(0)
		for value in parameter.grad.to_array() { grad_norm += value * value }
		assert grad_norm > 0, 'GRU backward must propagate gradients to each parameter'
	}
}

fn test_gru_layer_rejects_wrong_feature_size() {
	ctx := autograd.ctx[f64]()
	layer := gru_layer[f64](ctx, 4, 2)
	input := ctx.variable(vtl.zeros[f64]([3, 1, 5]))
	result := layer.forward(input) or { return }
	assert result.value.shape == [3, 1, 2], 'expected wrong input feature size to fail'
}
