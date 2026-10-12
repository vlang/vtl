module layers

import math
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

fn test_gru_layer_forward_with_state_returns_final_state_and_gradients() ! {
	ctx := autograd.ctx[f64]()
	layer := new_gru_layer[f64](ctx, 2, 3)
	input := ctx.variable(vtl.from_array([0.2, -0.1, 0.4, 0.3], [2, 1, 2])!)
	h0 := ctx.variable(vtl.from_array([0.1, -0.2, 0.3], [1, 3])!)
	mut output, mut final_state := layer.forward_with_state(input, h0)!
	assert output.value.shape == [2, 1, 3]
	assert final_state.value.shape == [1, 3]
	mut loss := output.sum()!.add(final_state.sum()!)!
	loss.backprop()!
	assert input.grad.shape == input.value.shape
	assert h0.grad.shape == h0.value.shape
	assert h0.grad.to_array().any(it != 0), 'GRU backward must propagate gradients to h0'
	for parameter in layer.variables() {
		assert parameter.grad.to_array().any(it != 0), 'GRU backward must propagate parameter gradients'
	}
}

fn test_gru_layer_batch_first_matches_sequence_first_and_backpropagates() ! {
	ctx := autograd.ctx[f64]()
	layer := new_gru_layer[f64](ctx, 2, 3)
	sequence_input := ctx.variable(vtl.from_array([0.2, -0.1, 0.4, 0.3, 0.5, 0.7, -0.2, 0.6], [
		2,
		2,
		2,
	])!)
	batch_input := ctx.variable(vtl.from_array([0.2, -0.1, 0.5, 0.7, 0.4, 0.3, -0.2, 0.6], [
		2,
		2,
		2,
	])!)
	sequence_output := layer.forward(sequence_input)!
	mut batch_output := layer.forward_batch_first(batch_input)!
	assert batch_output.value.shape == [2, 2, 3]
	sequence_values := sequence_output.value.to_array()
	batch_values := batch_output.value.to_array()
	for batch in 0 .. 2 {
		for timestep in 0 .. 2 {
			for feature in 0 .. 3 {
				batch_index := batch * 2 * 3 + timestep * 3 + feature
				sequence_index := timestep * 2 * 3 + batch * 3 + feature
				assert math.abs(batch_values[batch_index] - sequence_values[sequence_index]) < 1e-12
			}
		}
	}
	batch_output.sum()!.backprop()!
	assert batch_input.grad.shape == batch_input.value.shape
	assert batch_input.grad.to_array().any(it != 0)
}

fn test_gru_layer_batch_first_state_api_preserves_batch_shape_and_gradients() ! {
	ctx := autograd.ctx[f64]()
	layer := new_gru_layer[f64](ctx, 2, 3)
	input := ctx.variable(vtl.from_array([0.2, -0.1, 0.5, 0.7, 0.4, 0.3, -0.2, 0.6], [
		2,
		2,
		2,
	])!)
	h0 := ctx.variable(vtl.from_array([0.1, -0.2, 0.3, 0.0, 0.2, -0.1], [2, 3])!)
	mut output, mut final_state := layer.forward_batch_first_with_state(input, h0)!
	assert output.value.shape == [2, 2, 3]
	assert final_state.value.shape == [2, 3]
	output.sum()!.add(final_state.sum()!)!.backprop()!
	assert input.grad.shape == input.value.shape
	assert h0.grad.to_array().any(it != 0)
}

fn test_gru_layer_forward_with_state_rejects_invalid_state_shape() {
	ctx := autograd.ctx[f64]()
	layer := new_gru_layer[f64](ctx, 2, 3)
	input := ctx.variable(vtl.zeros[f64]([2, 1, 2]))
	h0 := ctx.variable(vtl.zeros[f64]([1, 2]))
	_ = layer.forward_with_state(input, h0) or { return }
	assert false, 'expected the initial hidden state shape to be validated'
}

fn test_gru_layer_rejects_wrong_feature_size() {
	ctx := autograd.ctx[f64]()
	layer := gru_layer[f64](ctx, 4, 2)
	input := ctx.variable(vtl.zeros[f64]([3, 1, 5]))
	result := layer.forward(input) or { return }
	assert result.value.shape == [3, 1, 2], 'expected wrong input feature size to fail'
}
