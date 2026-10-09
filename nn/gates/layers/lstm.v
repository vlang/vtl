module layers

import vtl
import vtl.autograd
import vtl.nn.internal

// LSTMGate stores per-layer inputs and parameters needed by CPU BPTT.
pub struct LSTMGate[T] {
pub:
	input            &autograd.Variable[T] = unsafe { nil }
	layer_inputs     []&vtl.Tensor[T]
	hidden0_by_layer []&vtl.Tensor[T]
	cell0_by_layer   []&vtl.Tensor[T]
	w_ih             []&autograd.Variable[T]
	w_hh             []&autograd.Variable[T]
	b_ih             []&autograd.Variable[T]
	b_hh             []&autograd.Variable[T]
	initial_hidden   &autograd.Variable[T] = unsafe { nil }
	initial_cell     &autograd.Variable[T] = unsafe { nil }
	state_output     int
	batch_first      bool
}

// lstm_gate creates an autograd operation for a stack of LSTM layers.
pub fn lstm_gate[T](input &autograd.Variable[T], layer_inputs []&vtl.Tensor[T],
	hidden0 &vtl.Tensor[T], cell0 &vtl.Tensor[T], w_ih []&autograd.Variable[T],
	w_hh []&autograd.Variable[T], b_ih []&autograd.Variable[T],
	b_hh []&autograd.Variable[T], batch_first bool) &LSTMGate[T] {
	return &LSTMGate[T]{
		input:            input
		layer_inputs:     layer_inputs.clone()
		hidden0_by_layer: []&vtl.Tensor[T]{len: layer_inputs.len, init: hidden0}
		cell0_by_layer:   []&vtl.Tensor[T]{len: layer_inputs.len, init: cell0}
		w_ih:             w_ih.clone()
		w_hh:             w_hh.clone()
		b_ih:             b_ih.clone()
		b_hh:             b_hh.clone()
		batch_first:      batch_first
	}
}

// lstm_gate_with_state creates an operation that differentiates explicit
// initial states and one of the sequence, final-hidden, or final-cell outputs.
pub fn lstm_gate_with_state[T](input &autograd.Variable[T], layer_inputs []&vtl.Tensor[T],
	hidden0_by_layer []&vtl.Tensor[T], cell0_by_layer []&vtl.Tensor[T],
	w_ih []&autograd.Variable[T], w_hh []&autograd.Variable[T], b_ih []&autograd.Variable[T],
	b_hh []&autograd.Variable[T], initial_hidden &autograd.Variable[T],
	initial_cell &autograd.Variable[T], batch_first bool, state_output int) &LSTMGate[T] {
	return &LSTMGate[T]{
		input:            input
		layer_inputs:     layer_inputs.clone()
		hidden0_by_layer: hidden0_by_layer.clone()
		cell0_by_layer:   cell0_by_layer.clone()
		w_ih:             w_ih.clone()
		w_hh:             w_hh.clone()
		b_ih:             b_ih.clone()
		b_hh:             b_hh.clone()
		initial_hidden:   initial_hidden
		initial_cell:     initial_cell
		state_output:     state_output
		batch_first:      batch_first
	}
}

// backward computes exact BPTT gradients for input and every layer parameter.
pub fn (g &LSTMGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	mut gradient := if g.state_output == 0 {
		payload.variable.grad
	} else {
		vtl.zeros[T]([g.layer_inputs[g.layer_inputs.len - 1].shape[0],
			g.layer_inputs[g.layer_inputs.len - 1].shape[1], g.w_hh[g.w_hh.len - 1].value.shape[1]])
	}
	if g.batch_first && g.state_output == 0 { gradient = gradient.transpose([1, 0, 2])! }
	mut param_gradients := [][]&vtl.Tensor[T]{len: g.layer_inputs.len}
	mut hidden0_gradients := []&vtl.Tensor[T]{len: g.layer_inputs.len}
	mut cell0_gradients := []&vtl.Tensor[T]{len: g.layer_inputs.len}
	for reverse_index in 0 .. g.layer_inputs.len {
		layer := g.layer_inputs.len - 1 - reverse_index
		mut grad_final_hidden := vtl.zeros[T](g.hidden0_by_layer[layer].shape)
		mut grad_final_cell := vtl.zeros[T](g.cell0_by_layer[layer].shape)
		if g.state_output == 1 {
			grad_final_hidden = tensor_layer_state[T](payload.variable.grad, layer)!
		}
		if g.state_output == 2 {
			grad_final_cell = tensor_layer_state[T](payload.variable.grad, layer)!
		}
		grads := internal.lstm_backward_single_with_final_state[T](g.layer_inputs[layer],
			g.hidden0_by_layer[layer], g.cell0_by_layer[layer], g.w_ih[layer].value,
			g.w_hh[layer].value, g.b_ih[layer].value, g.b_hh[layer].value, gradient,
			grad_final_hidden, grad_final_cell)!
		gradient = grads[0]
		param_gradients[layer] = [grads[1], grads[2], grads[3], grads[4]]
		hidden0_gradients[layer] = grads[5]
		cell0_gradients[layer] = grads[6]
	}
	if g.batch_first { gradient = gradient.transpose([1, 0, 2])! }
	mut result := [gradient]
	for layer in 0 .. g.layer_inputs.len {
		result << param_gradients[layer]
	}
	if g.initial_hidden != unsafe { nil } {
		result << stack_lstm_states[T](hidden0_gradients)!
		result << stack_lstm_states[T](cell0_gradients)!
	}
	return result
}

fn tensor_layer_state[T](tensor &vtl.Tensor[T], layer int) !&vtl.Tensor[T] {
	batch, hidden := tensor.shape[1], tensor.shape[2]
	mut values := []T{len: batch * hidden}
	for b in 0 .. batch {
		for h in 0 .. hidden {
			values[b * hidden + h] = tensor.get([layer, b, h])
		}
	}
	return vtl.from_array(values, [batch, hidden])
}

fn stack_lstm_states[T](states []&vtl.Tensor[T]) !&vtl.Tensor[T] {
	if states.len == 0 { return error('cannot stack empty LSTM states') }
	batch, hidden := states[0].shape[0], states[0].shape[1]
	mut values := []T{len: states.len * batch * hidden}
	for layer, state in states {
		for i in 0 .. batch * hidden {
			values[layer * batch * hidden + i] = state.get_nth(i)
		}
	}
	return vtl.from_array(values, [states.len, batch, hidden])
}

fn lstm_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&LSTMGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

// cache registers the sequence and all layer parameters in the autograd graph.
pub fn (g &LSTMGate[T]) cache(mut result autograd.Variable[T], _ ...autograd.CacheParam) ! {
	if g.layer_inputs.len == 0 || g.layer_inputs.len != g.w_ih.len || g.layer_inputs.len != g.w_hh.len
		|| g.layer_inputs.len != g.b_ih.len || g.layer_inputs.len != g.b_hh.len {
		return error('LSTMGate.cache: layer input and parameter counts must match')
	}
	if g.state_output < 0 || g.state_output > 2 || g.hidden0_by_layer.len != g.layer_inputs.len
		|| g.cell0_by_layer.len != g.layer_inputs.len {
		return error('LSTMGate.cache: invalid state output or initial-state count')
	}
	if (g.initial_hidden == unsafe { nil }) != (g.initial_cell == unsafe { nil }) {
		return error('LSTMGate.cache: initial hidden and cell variables must be supplied together')
	}
	result.grad = vtl.zeros_like[T](result.value)
	result.requires_grad = true
	mut parents := [g.input]
	for layer in 0 .. g.layer_inputs.len {
		parents << g.w_ih[layer]
		parents << g.w_hh[layer]
		parents << g.b_ih[layer]
		parents << g.b_hh[layer]
	}
	if g.initial_hidden != unsafe { nil } {
		parents << g.initial_hidden
		parents << g.initial_cell
	}
	autograd.register[T]('LSTM', voidptr(g), lstm_gate_backward_dispatch[T], result, parents)!
}
