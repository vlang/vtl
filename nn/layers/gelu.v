module layers

import vtl.autograd
import vtl.nn.internal
import vtl.nn.gates.activation
import vtl.nn.types

// GELULayer applies the Gaussian Error Linear Unit (GELU) activation.
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Approximation used: 0.7978845608 * (x + 0.044715 * x^3)  (lower-cost tanh approximation)
pub struct GELULayer[T] {
	output_shape []int
}

pub fn gelu_layer[T](ctx &autograd.Context[T], output_shape []int) types.Layer[T] {
	return types.Layer[T](&GELULayer[T]{
		output_shape: output_shape.clone()
	})
}

pub fn (layer &GELULayer[T]) output_shape() []int {
	return layer.output_shape
}

pub fn (_ &GELULayer[T]) variables() []&autograd.Variable[T] {
	return []&autograd.Variable[T]{}
}

pub fn (layer &GELULayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	if input.context.compute_strict && input.context.compute_backend != .cpu
		&& input.context.compute_backend != .auto {
		return error('gelu: backend `${input.context.compute_backend}` is not implemented for GPU path yet')
	}
	output := internal.gelu[T](input.value)
	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := activation.gelu_gate[T](input.value)
		gate.cache(mut result, input)!
	}
	return result
}
