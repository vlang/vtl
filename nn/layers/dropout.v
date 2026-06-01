module layers

import vtl
import vtl.autograd
import vtl.nn.internal
import vtl.nn.gates.layers
import vtl.nn.types

// DropoutLayerConfig defines a public data structure for this module.

// DropoutLayerConfig defines a public data structure for this module.
@[params]
pub struct DropoutLayerConfig {
	prob f64 = 0.5
}

// DropoutLayer is a dropout layer.
pub struct DropoutLayer[T] {
	output_shape []int
	prob         f64
}

// dropout_layer exposes this operation as part of the public API.
pub fn dropout_layer[T](ctx &autograd.Context[T], output_shape []int, data DropoutLayerConfig) types.Layer[T] {
	layer := &DropoutLayer[T]{
		output_shape: output_shape.clone()
		prob:         1.0 - data.prob
	}
	return types.layer[T](voidptr(layer), dropout_layer_output_shape_dispatch[T],
		dropout_layer_variables_dispatch[T], dropout_layer_forward_dispatch[T])
}

// output_shape exposes this operation as part of the public API.
pub fn (layer &DropoutLayer[T]) output_shape() []int {
	return layer.output_shape
}

// variables exposes this operation as part of the public API.
pub fn (_ &DropoutLayer[T]) variables() []&autograd.Variable[T] {
	return []&autograd.Variable[T]{}
}

// forward exposes this operation as part of the public API.
pub fn (layer &DropoutLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	mask := vtl.binomial[T](1, layer.prob, input.value.shape)!
	output := internal.dropout[T](input.value, mask, layer.prob)!
	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := layers.dropout_gate[T](mask, layer.prob)
		gate.cache(mut result, input)!
	}
	return result
}

fn dropout_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&DropoutLayer[T](layer)).output_shape() }
}

fn dropout_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&DropoutLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn dropout_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&DropoutLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}
