module layers

import vtl.nn.internal
import vtl.nn.types
import vtl.autograd
import vtl
import vtl.nn.gates.layers

// LSTMLayer implements a Long Short-Term Memory layer.
//
// Input:  `[batch, seq_len, input_size]`
// Output: `[batch, seq_len, hidden_size]`  (final hidden state per timestep)
//
// Contains learnable weights `w_ih`, `w_hh`, `b_ih`, `b_hh` for the input-to-hidden
// and hidden-to-hidden transformations of the four gates (input, forget, cell, output).
pub struct LSTMLayer[T] {
pub mut:
	w_ih        &autograd.Variable[T]
	w_hh        &autograd.Variable[T]
	b_ih        &autograd.Variable[T]
	b_hh        &autograd.Variable[T]
	w_ih_layers []&autograd.Variable[T]
	w_hh_layers []&autograd.Variable[T]
	b_ih_layers []&autograd.Variable[T]
	b_hh_layers []&autograd.Variable[T]
pub:
	ctx         &autograd.Context[T]
	input_size  int
	hidden_size int
	num_layers  int
}

// lstm_layer creates an LSTMLayer.
pub fn lstm_layer[T](ctx &autograd.Context[T], input_size int, hidden_size int, num_layers int) types.Layer[T] {
	return new_lstm_layer[T](ctx, input_size, hidden_size, num_layers).as_layer()
}

// new_lstm_layer creates an LSTM layer value with explicit state support.
// Use lstm_layer when composing the layer inside Sequential.
pub fn new_lstm_layer[T](ctx &autograd.Context[T], input_size int, hidden_size int, num_layers int) &LSTMLayer[T] {
	if input_size <= 0 || hidden_size <= 0 || num_layers <= 0 {
		panic('new_lstm_layer: input_size, hidden_size, and num_layers must be positive')
	}
	mut w_ih_layers := []&autograd.Variable[T]{cap: num_layers}
	mut w_hh_layers := []&autograd.Variable[T]{cap: num_layers}
	mut b_ih_layers := []&autograd.Variable[T]{cap: num_layers}
	mut b_hh_layers := []&autograd.Variable[T]{cap: num_layers}
	for index in 0 .. num_layers {
		layer_input_size := if index == 0 { input_size } else { hidden_size }
		w_ih_layers << ctx.variable(internal.kaiming_normal[T]([4 * hidden_size, layer_input_size]))
		w_hh_layers << ctx.variable(internal.kaiming_normal[T]([4 * hidden_size, hidden_size]))
		b_ih_layers << ctx.variable(vtl.zeros[T]([4 * hidden_size]))
		b_hh_layers << ctx.variable(vtl.zeros[T]([4 * hidden_size]))
	}
	mut layer := &LSTMLayer[T]{
		w_ih:        w_ih_layers[0]
		w_hh:        w_hh_layers[0]
		b_ih:        b_ih_layers[0]
		b_hh:        b_hh_layers[0]
		w_ih_layers: w_ih_layers
		w_hh_layers: w_hh_layers
		b_ih_layers: b_ih_layers
		b_hh_layers: b_hh_layers
		ctx:         ctx
		input_size:  input_size
		hidden_size: hidden_size
		num_layers:  num_layers
	}
	return layer
}

fn (l &LSTMLayer[T]) as_layer() types.Layer[T] {
	return types.layer[T](voidptr(l), lstm_layer_output_shape_dispatch[T],
		lstm_layer_variables_dispatch[T], lstm_layer_forward_dispatch[T])
}

fn (l &LSTMLayer[T]) output_shape() []int {
	return [l.hidden_size]
}

fn (l &LSTMLayer[T]) variables() []&autograd.Variable[T] {
	mut variables := []&autograd.Variable[T]{cap: 4 * l.num_layers}
	for index in 0 .. l.num_layers {
		variables << l.w_ih_layers[index]
		variables << l.w_hh_layers[index]
		variables << l.b_ih_layers[index]
		variables << l.b_hh_layers[index]
	}
	return variables
}

fn (l &LSTMLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	if input.context != l.ctx {
		return error('LSTMLayer.forward: input and layer must share an autograd context')
	}
	if input.value.shape.len != 3 || input.value.shape[2] != l.input_size {
		return error('LSTMLayer.forward: expected [batch, sequence, ${l.input_size}] input')
	}
	batch := input.value.shape[0]
	mut layer_input := input.value.transpose([1, 0, 2])!
	mut layer_inputs := []&vtl.Tensor[T]{cap: l.num_layers}
	hidden0 := vtl.zeros[T]([batch, l.hidden_size])
	cell0 := vtl.zeros[T]([batch, l.hidden_size])
	for index in 0 .. l.num_layers {
		layer_inputs << layer_input
		output_sequence, _, _ := internal.lstm_forward_single_with_cell[T](layer_input,
			hidden0, cell0, l.w_ih_layers[index].value, l.w_hh_layers[index].value,
			l.b_ih_layers[index].value, l.b_hh_layers[index].value)!
		layer_input = output_sequence
	}
	output := layer_input.transpose([1, 0, 2])!
	mut result := l.ctx.variable(output)
	variables := l.variables()
	if input.requires_grad || variables.any(it.requires_grad) {
		gate := layers.lstm_gate[T](input, layer_inputs, hidden0, cell0, l.w_ih_layers,
			l.w_hh_layers, l.b_ih_layers, l.b_hh_layers, true)
		gate.cache(mut result)!
	}
	return result
}

// forward_with_state accepts initial hidden and cell states with shape
// [num_layers, batch, hidden_size], then returns the output sequence and both
// final states in the same stacked shape.
pub fn (l &LSTMLayer[T]) forward_with_state(input &autograd.Variable[T],
	hidden0 &autograd.Variable[T], cell0 &autograd.Variable[T]) !(&autograd.Variable[T], &autograd.Variable[T], &autograd.Variable[T]) {
	if input.context != l.ctx || hidden0.context != l.ctx || cell0.context != l.ctx {
		return error('LSTMLayer.forward_with_state: input, states, and layer must share an autograd context')
	}
	if input.value.shape.len != 3 || input.value.shape[2] != l.input_size {
		return error('LSTMLayer.forward_with_state: expected [batch, sequence, ${l.input_size}] input')
	}
	batch := input.value.shape[0]
	expected_state_shape := [l.num_layers, batch, l.hidden_size]
	if hidden0.value.shape != expected_state_shape || cell0.value.shape != expected_state_shape {
		return error('LSTMLayer.forward_with_state: initial states must have shape ${expected_state_shape}')
	}
	mut layer_input := input.value.transpose([1, 0, 2])!
	mut layer_inputs := []&vtl.Tensor[T]{cap: l.num_layers}
	mut hidden0_by_layer := []&vtl.Tensor[T]{cap: l.num_layers}
	mut cell0_by_layer := []&vtl.Tensor[T]{cap: l.num_layers}
	mut final_hidden_by_layer := []&vtl.Tensor[T]{cap: l.num_layers}
	mut final_cell_by_layer := []&vtl.Tensor[T]{cap: l.num_layers}
	for index in 0 .. l.num_layers {
		layer_inputs << layer_input
		layer_hidden0 := lstm_state_for_layer[T](hidden0.value, index)!
		layer_cell0 := lstm_state_for_layer[T](cell0.value, index)!
		hidden0_by_layer << layer_hidden0
		cell0_by_layer << layer_cell0
		layer_input, layer_hidden, layer_cell := internal.lstm_forward_single_with_cell[T](layer_input, layer_hidden0, layer_cell0, l.w_ih_layers[index].value,
			l.w_hh_layers[index].value, l.b_ih_layers[index].value, l.b_hh_layers[index].value)!
		final_hidden_by_layer << layer_hidden
		final_cell_by_layer << layer_cell
	}
	output := layer_input.transpose([1, 0, 2])!
	final_hidden := stack_lstm_states[T](final_hidden_by_layer)!
	final_cell := stack_lstm_states[T](final_cell_by_layer)!
	mut output_result := l.ctx.variable(output)
	mut hidden_result := l.ctx.variable(final_hidden)
	mut cell_result := l.ctx.variable(final_cell)
	variables := l.variables()
	if input.is_grad_needed() || hidden0.is_grad_needed() || cell0.is_grad_needed()
		|| variables.any(it.is_grad_needed()) {
		output_gate := layers.lstm_gate_with_state[T](input, layer_inputs,
			hidden0_by_layer, cell0_by_layer, l.w_ih_layers, l.w_hh_layers, l.b_ih_layers,
			l.b_hh_layers, hidden0, cell0, true, 0)
		hidden_gate := layers.lstm_gate_with_state[T](input, layer_inputs, hidden0_by_layer,
			cell0_by_layer, l.w_ih_layers, l.w_hh_layers, l.b_ih_layers, l.b_hh_layers,
			hidden0, cell0, true, 1)
		cell_gate := layers.lstm_gate_with_state[T](input, layer_inputs, hidden0_by_layer,
			cell0_by_layer, l.w_ih_layers, l.w_hh_layers, l.b_ih_layers, l.b_hh_layers,
			hidden0, cell0, true, 2)
		output_gate.cache(mut output_result)!
		hidden_gate.cache(mut hidden_result)!
		cell_gate.cache(mut cell_result)!
	}
	return output_result, hidden_result, cell_result
}

fn lstm_state_for_layer[T](states &vtl.Tensor[T], layer int) !&vtl.Tensor[T] {
	batch, hidden := states.shape[1], states.shape[2]
	mut values := []T{len: batch * hidden}
	for b in 0 .. batch {
		for h in 0 .. hidden {
			values[b * hidden + h] = states.get([layer, b, h])
		}
	}
	return vtl.from_array(values, [batch, hidden])
}

fn lstm_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&LSTMLayer[T](layer)).output_shape() }
}

fn lstm_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&LSTMLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn lstm_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&LSTMLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}
