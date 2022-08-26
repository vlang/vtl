module layers

import vtl
import vtl.autograd
import vtl.nn.internal
import vtl.nn.gates.activation
import vtl.nn.types

[params]
pub struct EluLayerConfig {
	alpha f64 = 0.01
}

// EluLayer is an activation layer that applies the element-wise function
// `f(x) = x > 0 ? x : alpha * (exp(x) - 1)`
pub struct EluLayer<T> {
	output_shape []int
	alpha        f64
}

pub fn new_elu_layer<T>(ctx &autograd.Context<T>, output_shape []int, data EluLayerConfig) types.Layer {
	return types.Layer(&EluLayer<T>{
		output_shape: output_shape.clone()
		alpha: data.alpha
	})
}

pub fn (layer &EluLayer<T>) output_shape() []int {
	return layer.output_shape
}

pub fn (_ &EluLayer<T>) variables() []&autograd.Variable<T> {
	return []&autograd.Variable<T>{}
}

pub fn (layer &EluLayer<T>) forward(mut input autograd.Variable<T>) ?&autograd.Variable<T> {
	output := internal.elu<T>(input.value, layer.alpha)
	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := activation.new_elu_gate<T>(input.value)
		gate.cache(mut result, input)?
	}
	return result
}

pub fn elu<T>(v &autograd.Variable<T>, data EluLayerConfig) ?&autograd.Variable<T> {
	output := internal.elu<T>(v.value, layer.alpha)
	mut result := v.context.variable(output)

	if v.requires_grad {
		gate := activation.new_elu_gate<T>(v.value)
		gate.cache(mut result, v)?
	}
	return result
}
