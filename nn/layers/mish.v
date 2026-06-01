module layers

import vtl.autograd
import vtl.nn.internal
import vtl.nn.gates.activation
import vtl.nn.types

// MishLayer applies the Mish activation: x * tanh(softplus(x)).
// Mish(x) = x * tanh(softplus(x))  where softplus(x) = log(1 + exp(x))
pub struct MishLayer[T] {
	output_shape []int
}

// mish_layer exposes this operation as part of the public API.
pub fn mish_layer[T](ctx &autograd.Context[T], output_shape []int) types.Layer[T] {
	layer := &MishLayer[T]{
		output_shape: output_shape.clone()
	}
	return types.layer[T](voidptr(layer), mish_layer_output_shape_dispatch[T],
		mish_layer_variables_dispatch[T], mish_layer_forward_dispatch[T])
}

// output_shape exposes this operation as part of the public API.
pub fn (layer &MishLayer[T]) output_shape() []int {
	return layer.output_shape
}

// variables exposes this operation as part of the public API.
pub fn (_ &MishLayer[T]) variables() []&autograd.Variable[T] {
	return []&autograd.Variable[T]{}
}

// forward exposes this operation as part of the public API.
pub fn (layer &MishLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	output := internal.mish[T](input.value)
	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := activation.mish_gate[T](input.value)
		gate.cache(mut result, input)!
	}
	return result
}

fn mish_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&MishLayer[T](layer)).output_shape() }
}

fn mish_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&MishLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn mish_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&MishLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}
