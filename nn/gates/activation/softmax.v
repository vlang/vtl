module activation

import vtl
import vtl.autograd
import vtl.nn.internal

// SoftmaxGate stores the input tensor and dim to compute jacobian on backward.
pub struct SoftmaxGate[T] {
pub:
	input &vtl.Tensor[T] = unsafe { nil }
	dim   int
}

// softmax_gate exposes this operation as part of the public API.
pub fn softmax_gate[T](input &vtl.Tensor[T], dim int) &SoftmaxGate[T] {
	return &SoftmaxGate[T]{
		input: input
		dim:   dim
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &SoftmaxGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := internal.deriv_softmax[T](gradient, g.input, g.dim)!
	return [r0]
}

fn softmax_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&SoftmaxGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &SoftmaxGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			autograd.register[T]('Softmax', voidptr(g), softmax_gate_backward_dispatch[T], result, [
				a,
			])!
		}
		else {
			return error('Softmax: cache: invalid argument')
		}
	}
}
