module layers

import vtl
import vtl.autograd
import vtl.nn.internal

// LSTMGate stores per-layer inputs and parameters needed by CPU BPTT.
pub struct LSTMGate[T] {
pub:
	input        &autograd.Variable[T] = unsafe { nil }
	layer_inputs []&vtl.Tensor[T]
	hidden0      &vtl.Tensor[T] = unsafe { nil }
	cell0        &vtl.Tensor[T] = unsafe { nil }
	w_ih         []&autograd.Variable[T]
	w_hh         []&autograd.Variable[T]
	b_ih         []&autograd.Variable[T]
	b_hh         []&autograd.Variable[T]
	batch_first  bool
}

// lstm_gate creates an autograd operation for a stack of LSTM layers.
pub fn lstm_gate[T](input &autograd.Variable[T], layer_inputs []&vtl.Tensor[T],
	hidden0 &vtl.Tensor[T], cell0 &vtl.Tensor[T], w_ih []&autograd.Variable[T],
	w_hh []&autograd.Variable[T], b_ih []&autograd.Variable[T],
	b_hh []&autograd.Variable[T], batch_first bool) &LSTMGate[T] {
	return &LSTMGate[T]{
		input:        input
		layer_inputs: layer_inputs.clone()
		hidden0:      hidden0
		cell0:        cell0
		w_ih:         w_ih.clone()
		w_hh:         w_hh.clone()
		b_ih:         b_ih.clone()
		b_hh:         b_hh.clone()
		batch_first:  batch_first
	}
}

// backward computes exact BPTT gradients for input and every layer parameter.
pub fn (g &LSTMGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	mut gradient := payload.variable.grad
	if g.batch_first {
		gradient = gradient.transpose([1, 0, 2])!
	}
	mut param_gradients := [][]&vtl.Tensor[T]{len: g.layer_inputs.len}
	for reverse_index in 0 .. g.layer_inputs.len {
		layer := g.layer_inputs.len - 1 - reverse_index
		grads := internal.lstm_backward_single[T](g.layer_inputs[layer], g.hidden0, g.cell0,
			g.w_ih[layer].value, g.w_hh[layer].value, g.b_ih[layer].value,
			g.b_hh[layer].value, gradient)!
		gradient = grads[0]
		param_gradients[layer] = [grads[1], grads[2], grads[3], grads[4]]
	}
	if g.batch_first {
		gradient = gradient.transpose([1, 0, 2])!
	}
	mut result := [gradient]
	for layer in 0 .. g.layer_inputs.len {
		result << param_gradients[layer]
	}
	return result
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
	result.grad = vtl.zeros_like[T](result.value)
	result.requires_grad = true
	mut parents := [g.input]
	for layer in 0 .. g.layer_inputs.len {
		parents << g.w_ih[layer]
		parents << g.w_hh[layer]
		parents << g.b_ih[layer]
		parents << g.b_hh[layer]
	}
	autograd.register[T]('LSTM', voidptr(g), lstm_gate_backward_dispatch[T], result, parents)!
}
