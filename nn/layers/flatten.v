module layers

import vtl.autograd
import vtl.nn.gates.layers
import vtl.nn.types

// FlattenLayer is a layer
pub struct FlattenLayer[T] {
	shape []int
}

// flatten_layer exposes this operation as part of the public API.
pub fn flatten_layer[T](ctx &autograd.Context[T], shape []int) types.Layer[T] {
	layer := &FlattenLayer[T]{
		shape: shape.clone()
	}
	return types.layer[T](voidptr(layer), flatten_layer_output_shape_dispatch[T],
		flatten_layer_variables_dispatch[T], flatten_layer_forward_dispatch[T])
}

// output_shape exposes this operation as part of the public API.
pub fn (layer &FlattenLayer[T]) output_shape() []int {
	mut product := 1
	for s in layer.shape {
		if s > 0 {
			product *= s
		}
	}
	return [product]
}

// variables exposes this operation as part of the public API.
pub fn (_ &FlattenLayer[T]) variables() []&autograd.Variable[T] {
	return []&autograd.Variable[T]{}
}

// forward exposes this operation as part of the public API.
pub fn (layer &FlattenLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	output := input.value.reshape([input.value.shape[0], -1])!
	mut result := input.context.variable(output)

	if input.requires_grad {
		mut feat_shape := input.value.shape[1..].clone()
		gate := layers.flatten_gate[T](input, feat_shape)
		gate.cache(mut result, input)!
	}
	return result
}

fn flatten_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&FlattenLayer[T](layer)).output_shape() }
}

fn flatten_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&FlattenLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn flatten_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&FlattenLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}
