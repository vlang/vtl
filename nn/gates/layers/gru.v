module layers

import vtl
import vtl.autograd
import vtl.nn.internal

// GRUGate caches a GRU operation for reverse-mode differentiation.
pub struct GRUGate[T] {
pub:
	input              &autograd.Variable[T] = unsafe { nil }
	w_ih               &autograd.Variable[T] = unsafe { nil }
	w_hh               &autograd.Variable[T] = unsafe { nil }
	b_ih               &autograd.Variable[T] = unsafe { nil }
	b_hh               &autograd.Variable[T] = unsafe { nil }
	h0                 &autograd.Variable[T] = unsafe { nil }
	final_state_output bool
}

// gru_gate creates the autograd operation for one GRU layer.
pub fn gru_gate[T](input &autograd.Variable[T], w_ih &autograd.Variable[T], w_hh &autograd.Variable[T],
	b_ih &autograd.Variable[T], b_hh &autograd.Variable[T]) &GRUGate[T] {
	return &GRUGate[T]{ input: input, w_ih: w_ih, w_hh: w_hh, b_ih: b_ih, b_hh: b_hh }
}

// gru_gate_with_state creates a GRU gate that differentiates an explicit
// initial hidden state and optionally a gradient from the final hidden state.
pub fn gru_gate_with_state[T](input &autograd.Variable[T], w_ih &autograd.Variable[T],
	w_hh &autograd.Variable[T], b_ih &autograd.Variable[T], b_hh &autograd.Variable[T],
	h0 &autograd.Variable[T], final_state_output bool) &GRUGate[T] {
	return &GRUGate[T]{
		input:              input
		w_ih:               w_ih
		w_hh:               w_hh
		b_ih:               b_ih
		b_hh:               b_hh
		h0:                 h0
		final_state_output: final_state_output
	}
}

// backward returns derivatives for input and each trainable GRU parameter.
pub fn (g &GRUGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	input := g.input.value
	hidden := g.w_hh.value.shape[1]
	h0 := if g.h0 == unsafe { nil } { vtl.zeros[T]([input.shape[1], hidden]) } else { g.h0.value }
	mut grad_output := payload.variable.grad
	mut grad_final_state := vtl.zeros[T](h0.shape)
	if g.final_state_output {
		grad_final_state = payload.variable.grad
		grad_output = vtl.zeros[T](g.input.value.shape[..2] + [hidden])
	}
	gradients := internal.gru_backward_single_with_final_state[T](input, g.w_ih.value,
		g.w_hh.value, g.b_ih.value, g.b_hh.value, h0, grad_output, grad_final_state)!
	if g.h0 == unsafe { nil } {
		return gradients[..5]
	}
	return gradients
}

fn gru_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&GRUGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

// cache registers the GRU operation in its input's context.
pub fn (g &GRUGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	result.grad = vtl.zeros_like[T](result.value)
	result.requires_grad = true
	if g.h0 == unsafe { nil } {
		autograd.register[T]('GRU', voidptr(g), gru_gate_backward_dispatch[T], result,
			[g.input, g.w_ih, g.w_hh, g.b_ih, g.b_hh])!
	} else {
		autograd.register[T]('GRU', voidptr(g), gru_gate_backward_dispatch[T], result,
			[g.input, g.w_ih, g.w_hh, g.b_ih, g.b_hh, g.h0])!
	}
}
