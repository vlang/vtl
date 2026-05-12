module layers

import vtl
import vtl.autograd
import vtl.nn.internal
import vtl.nn.gates.activation
import vtl.nn.types

@[params]
pub struct LeakyReluLayerConfig {
	slope f64 = 0.01
}

// LeakyReluLayer is an activation layer that applies the leaky relu function to the input.
pub struct LeakyReluLayer[T] {
	output_shape []int
	slope        f64
}

pub fn leaky_relu_layer[T](ctx &autograd.Context[T], output_shape []int, data LeakyReluLayerConfig) types.Layer[T] {
	return types.Layer[T](&LeakyReluLayer[T]{
		output_shape: output_shape.clone()
		slope:        data.slope
	})
}

pub fn (layer &LeakyReluLayer[T]) output_shape() []int {
	return layer.output_shape
}

pub fn (_ &LeakyReluLayer[T]) variables() []&autograd.Variable[T] {
	return []&autograd.Variable[T]{}
}

pub fn (layer &LeakyReluLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	if input.context.compute_strict && input.context.compute_backend != .cpu
		&& input.context.compute_backend != .auto {
		return error('leaky_relu: backend `${input.context.compute_backend}` is not implemented for GPU path yet')
	}
	output := internal.leaky_relu[T](input.value, vtl.cast[T](layer.slope))
	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := activation.leaky_relu_gate[T](input.value, vtl.cast[T](layer.slope))
		gate.cache(mut result, input)!
	}
	return result
}
