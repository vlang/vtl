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
	return types.Layer[T](&LeakyReluLayer[T]{
		output_shape: output_shape.clone()
		slope:        data.slope
	})
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
