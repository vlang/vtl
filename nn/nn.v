module nn

import vtl.autograd
import vtl.nn.layers
import vtl.nn.optimizers

pub struct NeuralNetwork<T> {
pub mut:
	info &NeuralNetworkContainer<T>
}

pub fn new_nn<T>(ctx &autograd.Context<T>) &NeuralNetwork<T> {
	return &NeuralNetwork<T>{
		info: new_nnc<T>(ctx)
	}
}

// input adds a new input layer to the network
// with the given shape.
pub fn (mut nn NeuralNetwork<T>) input(shape []int) {
	nn.info.input(shape)
}

// linear adds a new linear layer to the network
// with the given output size
pub fn (mut nn NeuralNetwork<T>) linear(output_size int) {
	nn.info.linear(output_size)
}

// maxpool2d adds a new maxpool2d layer to the network
// with the given kernel size and stride.
pub fn (mut nn NeuralNetwork<T>) maxpool2d(kernel []int, padding []int, stride []int) {
	nn.info.maxpool2d(kernel, padding, stride)
}

// @todo: softmax_cross_entropy_loss
pub fn (mut nn NeuralNetwork<T>) softmax_cross_entropy_loss() {
	nn.info.softmax_cross_entropy_loss()
}

// flatten adds a new flatten layer to the network.
pub fn (mut nn NeuralNetwork<T>) flatten() {
	nn.info.flatten()
}

// relu adds a new relu layer to the network.
pub fn (mut nn NeuralNetwork<T>) relu() {
	nn.info.relu()
}

// leaky_relu adds a new leaky_relu layer to the network.
pub fn (mut nn NeuralNetwork<T>) leaky_relu() {
	nn.info.leaky_relu()
}

// elu adds a new elu layer to the network.
pub fn (mut nn NeuralNetwork<T>) elu() {
	nn.info.elu()
}

// sigmod adds a new sigmod layer to the network.
pub fn (mut nn NeuralNetwork<T>) sigmod() {
	nn.info.sigmod()
}

// sgd adds a new sgd optimizer to the network.
pub fn (mut nn NeuralNetwork<T>) sgd(config optimizers.SgdOptimizerConfig) {
	nn.info.sgd(config)
}

// adam adds a new adam optimizer to the network.
pub fn (mut nn NeuralNetwork<T>) adam(config optimizers.AdamOptimizerConfig) {
	nn.info.adam(config)
}

pub fn (mut nn NeuralNetwork<T>) forward(train &autograd.Variable<T>) ?&autograd.Variable<T> {
	mut ret := unsafe { train }
	for layer in nn.info.layers {
		ret = layers.layer_forward<T>(layer, mut ret)?
	}
	return ret
}
