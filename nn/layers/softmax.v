module layers

import vtl.autograd
import vtl.nn.internal
import vtl.nn.types
import vtl.nn.gates.activation as activation_gates

// SoftmaxLayer applies softmax activation over the last dimension of the input.
// input shape: [..., n_classes]  →  output shape: [..., n_classes]
// Softmax output sums to 1 along the last dimension.
pub struct SoftmaxLayer[T] {
	dim int
}

// SoftmaxLayerConfig defines a public data structure for this module.

// SoftmaxLayerConfig defines a public data structure for this module.
@[params]
pub struct SoftmaxLayerConfig {
	dim int = -1 // dimension to apply softmax over; -1 means last dimension
}

// softmax_layer exposes this operation as part of the public API.
pub fn softmax_layer[T](ctx &autograd.Context[T], config SoftmaxLayerConfig) types.Layer[T] {
	layer := &SoftmaxLayer[T]{
		dim: config.dim
	}
	return types.layer[T](voidptr(layer), softmax_layer_output_shape_dispatch[T],
		softmax_layer_variables_dispatch[T], softmax_layer_forward_dispatch[T])
}

// output_shape exposes this operation as part of the public API.
pub fn (layer &SoftmaxLayer[T]) output_shape() []int {
	return []int{}
}

// variables exposes this operation as part of the public API.
pub fn (layer &SoftmaxLayer[T]) variables() []&autograd.Variable[T] {
	return []&autograd.Variable[T]{}
}

// forward exposes this operation as part of the public API.
pub fn (layer &SoftmaxLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	// Compute softmax along dim (-1 means last dim)
	dim := if layer.dim == -1 { input.value.shape.len - 1 } else { layer.dim }
	output := internal.softmax_forward[T](input.value, dim)!
	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := activation_gates.softmax_gate[T](input.value, dim)
		gate.cache(mut result, input)!
	}
	return result
}

fn softmax_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&SoftmaxLayer[T](layer)).output_shape() }
}

fn softmax_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&SoftmaxLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn softmax_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&SoftmaxLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}
