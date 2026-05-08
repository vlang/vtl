module layers

import vtl.autograd
import vtl.nn.internal
import vtl.nn.gates.activation
import vtl.nn.types

// SwishLayer applies the Swish activation: x * sigmoid(x).
// Swish(x) = x * sigmoid(beta * x)  — here we use beta=1 (standard Swish).
pub struct SwishLayer[T] {
	output_shape []int
}

pub fn swish_layer[T](ctx &autograd.Context[T], output_shape []int) types.Layer[T] {
	return types.Layer[T](&SwishLayer[T]{
		output_shape: output_shape.clone()
	})
}

pub fn (layer &SwishLayer[T]) output_shape() []int {
	return layer.output_shape
}

pub fn (_ &SwishLayer[T]) variables() []&autograd.Variable[T] {
	return []&autograd.Variable[T]{}
}

pub fn (layer &SwishLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	output := internal.swish[T](input.value)
	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := activation.swish_gate[T](input.value)
		gate.cache(mut result, input)!
	}
	return result
}