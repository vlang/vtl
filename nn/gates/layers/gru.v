module layers

import vtl
import vtl.autograd
import vtl.nn.internal

// GRUGate caches a GRU operation for reverse-mode differentiation.
pub struct GRUGate[T] {
pub:
	input &autograd.Variable[T] = unsafe { nil }
	w_ih  &autograd.Variable[T] = unsafe { nil }
	w_hh  &autograd.Variable[T] = unsafe { nil }
	b_ih  &autograd.Variable[T] = unsafe { nil }
	b_hh  &autograd.Variable[T] = unsafe { nil }
}

// gru_gate creates the autograd operation for one GRU layer.
pub fn gru_gate[T](input &autograd.Variable[T], w_ih &autograd.Variable[T], w_hh &autograd.Variable[T],
	b_ih &autograd.Variable[T], b_hh &autograd.Variable[T]) &GRUGate[T] {
	return &GRUGate[T]{ input: input, w_ih: w_ih, w_hh: w_hh, b_ih: b_ih, b_hh: b_hh }
}

// backward returns derivatives for input and each trainable GRU parameter.
pub fn (g &GRUGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	input := g.input.value
	hidden := g.w_hh.value.shape[1]
	h0 := vtl.zeros[T]([input.shape[1], hidden])
	return internal.gru_backward_single[T](input, g.w_ih.value, g.w_hh.value, g.b_ih.value,
		g.b_hh.value, h0, payload.variable.grad)!
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
	autograd.register[T]('GRU', voidptr(g), gru_gate_backward_dispatch[T], result,
		[g.input, g.w_ih, g.w_hh, g.b_ih, g.b_hh])!
}
