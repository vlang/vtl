module layers

import vtl
import vtl.autograd
import vtl.nn.internal
import vtl.nn.gates.layers

[params]
pub struct DropoutLayerConfig {
        prob f64 = 0.5
}

// DropoutLayer is a dropout layer.
pub struct DropoutLayer<T> {
	output_shape []int
        prob f64
}

pub fn new_dropout_layer<T>(ctx &autograd.Context<T>, output_shape []int, data DropoutLayerConfig) &DropoutLayer<T> {
	return &DropoutLayer<T>{
		output_shape: output_shape.clone()
                prob: 1.0 - data.prob
	}
}

pub fn (layer &DropoutLayer<T>) output_shape() []int {
	return layer.output_shape
}

pub fn (_ &DropoutLayer<T>) variables() []&autograd.Variable<T> {
	return []&autograd.Variable<T>{}
}

pub fn (layer &DropoutLayer<T>) forward(mut input autograd.Variable<T>) &autograd.Variable<T> {
	mask := vtl.binomial<T>(1, layer.prob, input.value.shape)
        output := internal.dropout<T>(input.value, mask, layer.prob)
        mut result := input.context.variable(output)

        if input.requires_grad {
		gate := layers.new_dropout_gate<T>(mask, layer.prob)
		gate.cache(mut result, input)
	}
	return result
}
