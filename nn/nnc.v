module nn

import vtl.autograd
import vtl.nn.layers
import vtl.nn.loss
import vtl.nn.optimizers
import vtl.nn.types

pub struct NeuralNetworkContainer<T> {
	ctx &autograd.Context<T>
pub mut:
	layers    []types.Layer
	loss      types.Loss
	optimizer types.Optimizer
}

// new_nnc creates a new neural network container
// with an empty list of layers.
pub fn new_nnc<T>(ctx &autograd.Context<T>) &NeuralNetworkContainer<T> {
	return &NeuralNetworkContainer<T>{
		ctx: ctx
		loss: unsafe { nil }
		optimizer: unsafe { nil }
	}
}

// input adds a new input layer to the network
// with the given shape.
pub fn (mut ls NeuralNetworkContainer<T>) input(shape []int) {
	ls.layers << layers.new_input_layer<T>(ls.ctx, shape)
}

// linear adds a new linear layer to the network
// with the given output size
pub fn (mut ls NeuralNetworkContainer<T>) linear(output_size int) {
	input_size := layers.layer_output_shape<T>(ls.layers[ls.layers.len - 1])[0]
	ls.layers << layers.new_linear_layer<T>(ls.ctx, input_size, output_size)
}

// maxpool2d adds a new maxpool2d layer to the network
// with the given kernel size and stride.
pub fn (mut ls NeuralNetworkContainer<T>) maxpool2d(kernel []int, padding []int, stride []int) {
	shape := layers.layer_output_shape<T>(ls.layers[ls.layers.len - 1])
	ls.layers << layers.new_maxpool2d_layer<T>(ls.ctx, shape, kernel, padding, stride)
}

// @todo: mse_loss
pub fn (mut ls NeuralNetworkContainer<T>) mse_loss() {
	ls.loss = loss.new_mse_loss<T>()
}

// @todo: softmax_cross_entropy_loss
pub fn (mut ls NeuralNetworkContainer<T>) softmax_cross_entropy_loss() {
	// ls.loss =
}

// flatten adds a new flatten layer to the network.
pub fn (mut ls NeuralNetworkContainer<T>) flatten() {
	shape := layers.layer_output_shape<T>(ls.layers[ls.layers.len - 1])
	ls.layers << layers.new_flatten_layer<T>(ls.ctx, shape)
}

// relu adds a new relu layer to the network.
pub fn (mut ls NeuralNetworkContainer<T>) relu() {
	shape := layers.layer_output_shape<T>(ls.layers[ls.layers.len - 1])
	ls.layers << layers.new_relu_layer<T>(ls.ctx, shape)
}

// leaky_relu adds a new leaky_relu layer to the network.
pub fn (mut ls NeuralNetworkContainer<T>) leaky_relu() {
	shape := layers.layer_output_shape<T>(ls.layers[ls.layers.len - 1])
	ls.layers << layers.new_leaky_relu_layer<T>(ls.ctx, shape)
}

// elu adds a new elu layer to the network.
pub fn (mut ls NeuralNetworkContainer<T>) elu() {
	shape := layers.layer_output_shape<T>(ls.layers[ls.layers.len - 1])
	ls.layers << layers.new_elu_layer<T>(ls.ctx, shape)
}

// sigmod adds a new sigmod layer to the network.
pub fn (mut ls NeuralNetworkContainer<T>) sigmod() {
	shape := layers.layer_output_shape<T>(ls.layers[ls.layers.len - 1])
	ls.layers << layers.new_sigmoid_layer<T>(ls.ctx, shape)
}

// sgd adds a new sgd optimizer to the network.
pub fn (mut ls NeuralNetworkContainer<T>) sgd(config optimizers.SgdOptimizerConfig) {
	ls.optimizer = optimizers.new_sgd_optimizer<T>(config)
}

// adam adds a new adam optimizer to the network.
pub fn (mut ls NeuralNetworkContainer<T>) adam(config optimizers.AdamOptimizerConfig) {
	ls.optimizer = optimizers.new_adam_optimizer<T>(config)
}
