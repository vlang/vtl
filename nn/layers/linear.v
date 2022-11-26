module layers

import vtl
import vtl.la
import vtl.autograd
import vtl.nn.internal
import vtl.nn.gates.layers
import vtl.nn.types

// LinearLayer is a layer that applies a linear transformation to its input.
pub struct LinearLayer[T] {
	weights &autograd.Variable[T]
	bias    &autograd.Variable[T]
}

pub fn linear_layer[T](ctx &autograd.Context[T], input_dim int, output_dim int) types.Layer {
	weights := internal.kaiming_normal[T]([output_dim, input_dim])
	bias := vtl.zeros[T]([1, output_dim])
	return types.Layer(&LinearLayer[T]{
		weights: ctx.variable(weights)
		bias: ctx.variable(bias)
	})
}

pub fn (layer &LinearLayer[T]) output_shape() []int {
	return [layer.weights.value.shape[0]]
}

pub fn (layer &LinearLayer[T]) variables() []&autograd.Variable[T] {
	return [layer.weights, layer.bias]
}

pub fn (layer &LinearLayer[T]) forward(mut input autograd.Variable[T]) ?&autograd.Variable[T] {
	mut output := la.matmul[T](input.value, layer.weights.value)?.add[T](layer.bias.value)?
	mut result := input.context.variable(output)

	if input.requires_grad || layer.weights.requires_grad || layer.bias.requires_grad {
		gate := layers.linear_gate[T](input, layer.weights, layer.bias)
		gate.cache(mut result, input, layer.weights, layer.bias)?
	}

	return result
}
