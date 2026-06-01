module layers

import vtl.autograd
import vtl.nn.internal
import vtl.nn.gates.activation
import vtl.nn.types

// TanhLayer is a layer that applies the tanh activation function to its input.
pub struct TanhLayer[T] {
	output_shape []int
}

// tanh_layer exposes this operation as part of the public API.
pub fn tanh_layer[T](ctx &autograd.Context[T], output_shape []int) types.Layer[T] {
	layer := &TanhLayer[T]{
		output_shape: output_shape.clone()
	}
	return types.layer[T](voidptr(layer), tanh_layer_output_shape_dispatch[T],
		tanh_layer_variables_dispatch[T], tanh_layer_forward_dispatch[T])
}

// output_shape exposes this operation as part of the public API.
pub fn (layer &TanhLayer[T]) output_shape() []int {
	return layer.output_shape
}

// variables exposes this operation as part of the public API.
pub fn (_ &TanhLayer[T]) variables() []&autograd.Variable[T] {
	return []&autograd.Variable[T]{}
}

// forward exposes this operation as part of the public API.
pub fn (layer &TanhLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	output := internal.tanh[T](input.value)
	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := activation.tanh_gate[T](input.value)
		gate.cache(mut result, input)!
	}
	return result
}

fn tanh_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&TanhLayer[T](layer)).output_shape() }
}

fn tanh_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&TanhLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn tanh_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&TanhLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}
