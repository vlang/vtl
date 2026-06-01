module layers

import vtl.autograd
import vtl.nn.internal
import vtl.nn.gates.activation
import vtl.nn.types

// SwishLayer applies the Swish activation: x * sigmoid(x).
// Swish(x) = x * sigmoid(beta * x)  — here we use beta=1 (standard Swish).
pub struct SwishLayer[T] {
	output_shape []int
}

// swish_layer exposes this operation as part of the public API.
pub fn swish_layer[T](ctx &autograd.Context[T], output_shape []int) types.Layer[T] {
	layer := &SwishLayer[T]{
		output_shape: output_shape.clone()
	}
	return types.layer[T](voidptr(layer), swish_layer_output_shape_dispatch[T],
		swish_layer_variables_dispatch[T], swish_layer_forward_dispatch[T])
}

// output_shape exposes this operation as part of the public API.
pub fn (layer &SwishLayer[T]) output_shape() []int {
	return layer.output_shape
}

// variables exposes this operation as part of the public API.
pub fn (_ &SwishLayer[T]) variables() []&autograd.Variable[T] {
	return []&autograd.Variable[T]{}
}

// forward exposes this operation as part of the public API.
pub fn (layer &SwishLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	output := internal.swish[T](input.value)
	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := activation.swish_gate[T](input.value)
		gate.cache(mut result, input)!
	}
	return result
}

fn swish_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&SwishLayer[T](layer)).output_shape() }
}

fn swish_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&SwishLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn swish_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&SwishLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}
