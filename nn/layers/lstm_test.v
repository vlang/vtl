module layers

import vtl
import vtl.autograd

fn test_lstm_layer_batch_first_forward_and_autograd() ! {
	ctx := autograd.ctx[f64]()
	layer := lstm_layer[f64](ctx, 2, 3, 2)
	assert layer.output_shape() == [3]
	assert layer.variables().len == 8
	input := ctx.variable(vtl.from_array([0.2, -0.1, 0.4, 0.3, -0.2, 0.5, 0.1, -0.3, 0.6, 0.2,
		-0.4, 0.5], [2, 3, 2])!)
	output := layer.forward(input)!
	assert output.value.shape == [2, 3, 3]
	output.backprop()!
	assert input.grad.shape == [2, 3, 2]
	assert input.grad.to_array().any(it != 0)
	for parameter in layer.variables() {
		assert parameter.grad.shape == parameter.value.shape
		assert parameter.grad.to_array().any(it != 0)
	}
}

fn test_lstm_layer_rejects_non_rank_three_input() {
	ctx := autograd.ctx[f64]()
	layer := lstm_layer[f64](ctx, 2, 3, 1)
	input := ctx.variable(vtl.zeros[f64]([2, 2]))
	if _ := layer.forward(input) {
		assert false, 'LSTM layer must reject inputs outside [batch, sequence, features]'
	} else {
		assert true
	}
}
