module layers

import vtl
import vtl.autograd
import vtl.nn.internal

// MaxPool2DGate defines a public data structure for this module.
pub struct MaxPool2DGate[T] {
pub:
	max_indices &vtl.Tensor[int] = unsafe { nil }
	kernel      []int
	shape       []int
	stride      []int
	padding     []int
	// input is the forward input. It is nil unless the gate was built by
	// maxpool2d_gate_with_input, and cuDNN backward needs it.
	input &vtl.Tensor[T] = unsafe { nil }
}

// maxpool2d_gate exposes this operation as part of the public API.
pub fn maxpool2d_gate[T](max_indices &vtl.Tensor[int], kernel []int, shape []int, stride []int, padding []int) &MaxPool2DGate[T] {
	return &MaxPool2DGate[T]{
		max_indices: max_indices
		kernel:      kernel
		shape:       shape
		stride:      stride
		padding:     padding
	}
}

// maxpool2d_gate_with_input creates a gate that also keeps the forward input for cuDNN backward.
pub fn maxpool2d_gate_with_input[T](max_indices &vtl.Tensor[int], input &vtl.Tensor[T], kernel []int, shape []int, stride []int, padding []int) &MaxPool2DGate[T] {
	return &MaxPool2DGate[T]{
		max_indices: max_indices
		input:       input
		kernel:      kernel
		shape:       shape
		stride:      stride
		padding:     padding
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &MaxPool2DGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad

	if g.input != unsafe { nil } {
		r_input := internal.maxpool2d_backward_with_input[T](g.shape, g.max_indices, g.input,
			gradient, g.kernel, g.stride, g.padding)!
		return [r_input]
	}
	r0 := internal.maxpool2d_backward[T](g.shape, g.max_indices, gradient)!

	return [r0]
}

fn max_pool2_d_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&MaxPool2DGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &MaxPool2DGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]

	match a {
		autograd.Variable[T] {
			autograd.register[T]('MaxPool2D', voidptr(g), max_pool2_d_gate_backward_dispatch[T],
				result, [a])!
		}
		else {
			return error('MaxPool2DGate: cache: invalid argument')
		}
	}
}
