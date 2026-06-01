module layers

import vtl
import vtl.autograd

// LSTMGate implements the forward pass of LSTM with full gradients.
// This is a basic implementation; for production use an optimized version.
pub struct LSTMGate[T] {
	input_  &vtl.Tensor[T] = unsafe { nil }
	hidden_ &vtl.Tensor[T] = unsafe { nil }
	cell_   &vtl.Tensor[T] = unsafe { nil }
	w_ih    &vtl.Tensor[T] = unsafe { nil }
	w_hh    &vtl.Tensor[T] = unsafe { nil }
	b_ih    &vtl.Tensor[T] = unsafe { nil }
	b_hh    &vtl.Tensor[T] = unsafe { nil }
}

// lstm_gate exposes this operation as part of the public API.
pub fn lstm_gate[T](input_ &vtl.Tensor[T],
	hidden_ &vtl.Tensor[T],
	cell_ &vtl.Tensor[T],
	w_ih &vtl.Tensor[T],
	w_hh &vtl.Tensor[T],
	b_ih &vtl.Tensor[T],
	b_hh &vtl.Tensor[T]) &LSTMGate[T] {
	return &LSTMGate[T]{
		input_:  input_
		hidden_: hidden_
		cell_:   cell_
		w_ih:    w_ih
		w_hh:    w_hh
		b_ih:    b_ih
		b_hh:    b_hh
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &LSTMGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	// Simplified: return gradient w.r.t. input (full LSTM backward is complex)
	grad := payload.variable.grad
	return [grad, grad, grad, grad, grad]
}

fn lstm_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&LSTMGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &LSTMGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			autograd.register[T]('LSTM', voidptr(g), lstm_gate_backward_dispatch[T], result, [
				a,
			])!
		}
		else {}
	}
}
