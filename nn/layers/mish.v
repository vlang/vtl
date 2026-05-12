module layers

import vtl.autograd
import vtl.nn.internal
import vtl.nn.gates.activation
import vtl.nn.types

// MishLayer applies the Mish activation: x * tanh(softplus(x)).
// Mish(x) = x * tanh(softplus(x))  where softplus(x) = log(1 + exp(x))
pub struct MishLayer[T] {
	output_shape []int
}

pub fn mish_layer[T](ctx &autograd.Context[T], output_shape []int) types.Layer[T] {
	return types.Layer[T](&MishLayer[T]{
		output_shape: output_shape.clone()
	})
}

pub fn (layer &MishLayer[T]) output_shape() []int {
	return layer.output_shape
}

pub fn (_ &MishLayer[T]) variables() []&autograd.Variable[T] {
	return []&autograd.Variable[T]{}
}

pub fn (layer &MishLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	if input.context.compute_strict && input.context.compute_backend != .cpu
		&& input.context.compute_backend != .auto {
		return error('mish: backend `${input.context.compute_backend}` is not implemented for GPU path yet')
	}
	output := internal.mish[T](input.value)
	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := activation.mish_gate[T](input.value)
		gate.cache(mut result, input)!
	}
	return result
}
