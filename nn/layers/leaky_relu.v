module layers

import vtl
import vtl.autograd
import vtl.nn.internal
import vtl.nn.gates.activation
import vtl.nn.types

// LeakyReluLayerConfig defines a public data structure for this module.

// LeakyReluLayerConfig defines a public data structure for this module.
@[params]
pub struct LeakyReluLayerConfig {
	slope f64 = 0.01
}

// LeakyReluLayer is an activation layer that applies the leaky relu function to the input.
pub struct LeakyReluLayer[T] {
	output_shape []int
	slope        f64
}

// leaky_relu_layer exposes this operation as part of the public API.
pub fn leaky_relu_layer[T](ctx &autograd.Context[T], output_shape []int, data LeakyReluLayerConfig) types.Layer[T] {
	layer := &LeakyReluLayer[T]{
		output_shape: output_shape.clone()
		slope:        data.slope
	}
	return types.layer[T](voidptr(layer), leaky_relu_layer_output_shape_dispatch[T],
		leaky_relu_layer_variables_dispatch[T], leaky_relu_layer_forward_dispatch[T])
}

// output_shape exposes this operation as part of the public API.
pub fn (layer &LeakyReluLayer[T]) output_shape() []int {
	return layer.output_shape
}

// variables exposes this operation as part of the public API.
pub fn (_ &LeakyReluLayer[T]) variables() []&autograd.Variable[T] {
	return []&autograd.Variable[T]{}
}

// forward exposes this operation as part of the public API.
pub fn (layer &LeakyReluLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	output := internal.leaky_relu[T](input.value, vtl.cast[T](layer.slope))
	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := activation.leaky_relu_gate[T](input.value, vtl.cast[T](layer.slope))
		gate.cache(mut result, input)!
	}
	return result
}

fn leaky_relu_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&LeakyReluLayer[T](layer)).output_shape() }
}

fn leaky_relu_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&LeakyReluLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn leaky_relu_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&LeakyReluLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}
