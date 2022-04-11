module nn

import vtl.autograd
import vtl.nn.layers

pub struct NeuralNetworkContainer<T> {
	ctx &autograd.Context<T>
pub mut:
	layers []Layer
	loss   Loss
}

// new_nnc creates a new neural network container
// with an empty list of layers.
pub fn new_nnc<T>(ctx &autograd.Context<T>) &NeuralNetworkContainer<T> {
	return &NeuralNetworkContainer<T>{
		ctx: ctx
	}
}

// input adds a new input layer to the network
// with the given shape.
pub fn (mut ls NeuralNetworkContainer<T>) input(shape []int) {
	ls.layers << layers.new_input_layer(ls.ctx, shape)
}

// linear adds a new linear layer to the network
// with the given output size
pub fn (mut ls NeuralNetworkContainer<T>) linear(output_size int) {
	// input_size := ls.layers[ls.layers.len - 1].output_shape()[0]
	input_size := output_size
	ls.layers << layers.new_linear_layer<T>(ls.ctx, input_size, output_size)
}

// maxpool2d adds a new maxpool2d layer to the network
// with the given kernel size and stride.
pub fn (mut ls NeuralNetworkContainer<T>) maxpool2d(kernel []int, padding []int, stride []int) {
	// shape := ls.layers[ls.layers.len - 1].output_shape()
	shape := []int{}
	ls.layers << layers.new_maxpool2d_layer<T>(ls.ctx, shape, kernel, padding, stride)
}
