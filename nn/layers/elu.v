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
	return types.Layer[T](&EluLayer[T]{
		output_shape: output_shape.clone()
		alpha:        data.alpha
	})
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
