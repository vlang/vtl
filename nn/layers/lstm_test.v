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

fn test_lstm_layer_forward_with_state_returns_final_states_and_gradients() ! {
	ctx := autograd.ctx[f64]()
	layer := new_lstm_layer[f64](ctx, 2, 3, 2)
	input := ctx.variable(vtl.from_array([0.2, -0.1, 0.4, 0.3], [1, 2, 2])!)
	hidden0 := ctx.variable(vtl.from_array([0.1, -0.2, 0.3, 0.2, -0.1, 0.05], [2, 1, 3])!)
	cell0 := ctx.variable(vtl.from_array([-0.1, 0.2, 0.05, 0.1, -0.2, 0.3], [2, 1, 3])!)
	mut output, mut final_hidden, mut final_cell := layer.forward_with_state(input, hidden0,
		cell0)!
	assert output.value.shape == [1, 2, 3]
	assert final_hidden.value.shape == [2, 1, 3]
	assert final_cell.value.shape == [2, 1, 3]
	mut loss := output.sum()!.add(final_hidden.sum()!)!.add(final_cell.sum()!)!
	loss.backprop()!
	assert input.grad.shape == input.value.shape
	assert hidden0.grad.shape == hidden0.value.shape
	assert cell0.grad.shape == cell0.value.shape
	assert hidden0.grad.to_array().any(it != 0)
	assert cell0.grad.to_array().any(it != 0)
	for parameter in layer.variables() {
		assert parameter.grad.to_array().any(it != 0)
	}
}

fn test_lstm_layer_forward_with_state_rejects_wrong_state_shape() {
	ctx := autograd.ctx[f64]()
	layer := new_lstm_layer[f64](ctx, 2, 3, 2)
	input := ctx.variable(vtl.zeros[f64]([1, 2, 2]))
	hidden0 := ctx.variable(vtl.zeros[f64]([1, 1, 3]))
	cell0 := ctx.variable(vtl.zeros[f64]([2, 1, 3]))
	_, _, _ := layer.forward_with_state(input, hidden0, cell0) or { return }
	assert false, 'expected the stacked initial states to be validated'
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
