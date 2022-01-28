module layers

import vtl
import vtl.la
import vtl.autograd
import vtl.nn.gates.layers

// LinearLayer is a layer that applies a linear transformation to its input.
pub struct LinearLayer<T> {
	weights &autograd.Variable<T>
	bias    &autograd.Variable<T>
}

pub fn new_linear_layer<T>(ctx &autograd.Context<T>, input_dim int, output_dim int) &LinearLayer<T> {
	// @todo: FIX THIS
	// weight := vtl.nn.kaiming_normal<T>(output_dim, input_dim)
	bias := vtl.zeros<T>([1, output_dim])
	weights := bias
	return &LinearLayer<T>{
		weights: ctx.variable(weights)
		bias: ctx.variable(bias)
	}
}

pub fn (layer LinearLayer<T>) output_shape() []int {
	return [layer.weights.value.shape[0]]
}

pub fn (layer LinearLayer<T>) variables() []&autograd.Variable<T> {
	return [layer.weights, layer.bias]
}

pub fn (layer LinearLayer<T>) forward(mut input autograd.Variable<T>) &autograd.Variable<T> {
	output := vtl.add(la.matmul(input.value, layer.weights.value), layer.bias.value)
	mut result := input.context.variable(output)

	if input.requires_grad || layer.weights.requires_grad || layer.bias.requires_grad {
		gate := layers.new_linear_gate<T>(input, layer.weights, layer.bias)
		gate.cache(mut result, input, layer.weights, layer.bias)
	}

	return result
}
