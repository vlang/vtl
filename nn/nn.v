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
pub fn (mut nncon NNLayers<T>) input(shape []int) {
	nncon.layers << layers.new_input_layer(nncon.ctx, shape)
}

// linear adds a new linear layer to the network
// with the given output size
// pub fn (mut nncon NNLayers<T>) linear(output_size int) {
//         input_size := nncon.layers[nncon.layers.len - 1].output_shape[0]
//         nncon.layers << layers.new_linear_layer(nncon.ctx, input_size, output_size)
// }
