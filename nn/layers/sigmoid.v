module layers

import vtl
import vtl.autograd
import vtl.nn.internal
import vtl.nn.gates.layers

// SigmoidLayer is a layer that applies the sigmoid function to its input.
pub struct SigmoidLayer<T> {
	output_shape []int
}

pub fn new_sigmoid_layer<T>(ctx &autograd.Context<T>, output_shape []int) &SigmoidLayer<T> {
	return &SigmoidLayer<T>{
		output_shape: output_shape.clone()
	}
}

pub fn (layer &SigmoidLayer<T>) output_shape() []int {
	return layer.output_shape
}

pub fn (_ &SigmoidLayer<T>) variables() []&autograd.Variable<T> {
	return []&autograd.Variable<T>{}
}

pub fn (layer &SigmoidLayer<T>) forward(mut input autograd.Variable<T>) &autograd.Variable<T> {
	output := internal.sigmoid<T>(input.value)
	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := layers.new_sigmoid_gate<T>(input.value)
		gate.cache(mut result, input)
	}
	return result
}

pub fn sigmoid<T>(v &autograd.Variable<T>) &autograd.Variable<T> {
	output := internal.sigmoid<T>(v.value)
	mut result := v.context.variable(output)

	if v.requires_grad {
		gate := layers.new_sigmoid_gate<T>(v.value)
		gate.cache(mut result, v)
	}
	return result
}
