module layers

import vtl.autograd
import vtl.nn.gates.layers
import vtl.nn.types

// InputLayer is a layer that takes a single input tensor and returns the same
// tensor.
//
// This layer is used as the first layer in a model.
pub struct InputLayer[T] {
	shape []int
}

// input_layer exposes this operation as part of the public API.
pub fn input_layer[T](ctx &autograd.Context[T], shape []int) types.Layer[T] {
	layer := &InputLayer[T]{
		shape: shape.clone()
	}
	return types.layer[T](voidptr(layer), input_layer_output_shape_dispatch[T],
		input_layer_variables_dispatch[T], input_layer_forward_dispatch[T])
}

// output_shape exposes this operation as part of the public API.
pub fn (layer &InputLayer[T]) output_shape() []int {
	return layer.shape
}

// variables exposes this operation as part of the public API.
pub fn (_ &InputLayer[T]) variables() []&autograd.Variable[T] {
	return []&autograd.Variable[T]{}
}

// forward exposes this operation as part of the public API.
pub fn (layer &InputLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	mut result := input.context.variable(input.value, requires_grad: input.requires_grad)
	if input.requires_grad {
		gate := layers.input_gate[T]()
		gate.cache(mut result, input)!
	}
	return result
}

fn input_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&InputLayer[T](layer)).output_shape() }
}

fn input_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&InputLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn input_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&InputLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}
