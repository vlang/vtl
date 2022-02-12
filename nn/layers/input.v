module layers

import vtl
import vtl.autograd
import vtl.nn.gates.layers

// InputLayer is a layer that takes a single input tensor and returns the same
// tensor.
//
// This layer is used as the first layer in a model.
pub struct InputLayer<T> {
	shape []int
}

pub fn new_input_layer<T>(ctx &autograd.Context<T>, shape []int) &InputLayer<T> {
	return &InputLayer<T>{
		shape: shape.clone()
	}
}

pub fn (layer &InputLayer<T>) output_shape() []int {
	return layer.shape
}

pub fn (_ &InputLayer<T>) variables() []&autograd.Variable<T> {
	return []&autograd.Variable<T>{}
}

pub fn (layer &InputLayer<T>) forward(mut input autograd.Variable<T>) &autograd.Variable<T> {
	if input.requires_grad {
		gate := layers.new_input_gate<T>()
		gate.cache(mut input, input)
	}
	return input
}
