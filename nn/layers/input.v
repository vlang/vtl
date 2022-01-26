module layers

import vtl
import vtl.autograd

// InputLayer is a layer that takes a single input tensor and returns the same
// tensor.
//
// This layer is used as the first layer in a model.
pub struct InputLayer<T> {
	shape []int
pub:
	output_shape []int
}

pub fn new_input_layer<T>(ctx &autograd.Context<T>, shape []int) &InputLayer<T> {
	return &InputLayer<T>{
		shape: shape.clone()
		output_shape: shape.clone()
	}
}
