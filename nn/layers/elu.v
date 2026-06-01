module layers

import vtl
import vtl.autograd
import vtl.nn.internal
import vtl.nn.gates.activation
import vtl.nn.types

// EluLayerConfig defines a public data structure for this module.

// EluLayerConfig defines a public data structure for this module.
@[params]
pub struct EluLayerConfig {
	alpha f64 = 0.01
}

// EluLayer is an activation layer that applies the element-wise function
// `f(x) = x > 0 ? x : alpha * (exp(x) - 1)`
pub struct EluLayer[T] {
	output_shape []int
	alpha        f64
}

// elu_layer exposes this operation as part of the public API.
pub fn elu_layer[T](ctx &autograd.Context[T], output_shape []int, data EluLayerConfig) types.Layer[T] {
	layer := &EluLayer[T]{
		output_shape: output_shape.clone()
		alpha:        data.alpha
	}
	return types.layer[T](voidptr(layer), elu_layer_output_shape_dispatch[T],
		elu_layer_variables_dispatch[T], elu_layer_forward_dispatch[T])
}

// output_shape exposes this operation as part of the public API.
pub fn (layer &EluLayer[T]) output_shape() []int {
	return layer.output_shape
}

// variables exposes this operation as part of the public API.
pub fn (_ &EluLayer[T]) variables() []&autograd.Variable[T] {
	return []&autograd.Variable[T]{}
}

// forward exposes this operation as part of the public API.
pub fn (layer &EluLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	output := internal.elu[T](input.value, vtl.cast[T](layer.alpha))
	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := activation.elu_gate[T](input.value, vtl.cast[T](layer.alpha))
		gate.cache(mut result, input)!
	}
	return result
}

fn elu_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&EluLayer[T](layer)).output_shape() }
}

fn elu_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&EluLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn elu_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&EluLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}
