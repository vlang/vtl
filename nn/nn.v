module nn

import vtl.autograd
import vtl.nn.layers

pub struct NNLayers<T> {
	ctx &autograd.Context<T>
pub mut:
	layers []Layer
}

// new_nn_layers creates a new neural network container
// with an empty list of layers.
pub fn new_nn_layers<T>(ctx &autograd.Context<T>) &NNLayers<T> {
	return &NNLayers<T>{
		ctx: ctx
	}
}

// input adds a new input layer to the network
// with the given shape.
pub fn (mut ls NNLayers<T>) input(shape []int) {
	ls.layers << layers.new_input_layer(ls.ctx, shape)
}

// linear adds a new linear layer to the network
// with the given output size
pub fn (mut ls NNLayers<T>) linear(output_size int) {
	// input_size := ls.layers[ls.layers.len - 1].output_shape()[0]
	input_size := output_size
	ls.layers << layers.new_linear_layer<T>(ls.ctx, input_size, output_size)
}
