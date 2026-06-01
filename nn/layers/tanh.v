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
	return types.Layer[T](&TanhLayer[T]{
		output_shape: output_shape.clone()
	})
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
