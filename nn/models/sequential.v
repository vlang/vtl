module models

import vtl
import vtl.autograd
import vtl.nn.layers
import vtl.nn.loss
import vtl.nn.optimizers
import vtl.nn.types

fn init() {
	println(@MOD + ' module is a WIP and not yet functional')
}

pub struct Sequential<T> {
pub mut:
	info &SequentialInfo<T>
}

[params]
pub struct SequentialParams {
	layers []types.Layer
}

pub fn sequential<T>(params SequentialParams) &Sequential<T> {
	ctx := autograd.new_ctx<T>()
	return &Sequential<T>{
		info: new_sequential_info<T>(ctx, params.layers)
	}
}

pub fn sequential_from_ctx<T>(ctx &autograd.Context<T>, params SequentialParams) &Sequential<T> {
	return &Sequential<T>{
		info: new_sequential_info<T>(ctx, params.layers)
	}
}

// input adds a new input layer to the network
// with the given shape.
pub fn (mut nn Sequential<T>) input(shape []int) {
	nn.info.input(shape)
}

// linear adds a new linear layer to the network
// with the given output size
pub fn (mut nn Sequential<T>) linear(output_size int) {
	nn.info.linear(output_size)
}

// maxpool2d adds a new maxpool2d layer to the network
// with the given kernel size and stride.
pub fn (mut nn Sequential<T>) maxpool2d(kernel []int, padding []int, stride []int) {
	nn.info.maxpool2d(kernel, padding, stride)
}

// mse_loss sets the loss function to the mean squared error loss.
pub fn (mut nn Sequential<T>) mse_loss() {
	nn.info.mse_loss()
}

// sigmoid_cross_entropy_loss sets the loss function to the sigmoid cross entropy loss.
pub fn (mut nn Sequential<T>) sigmoid_cross_entropy_loss() {
	nn.info.sigmoid_cross_entropy_loss()
}

// softmax_cross_entropy_loss sets the loss function to the softmax cross entropy loss.
pub fn (mut nn Sequential<T>) softmax_cross_entropy_loss() {
	nn.info.softmax_cross_entropy_loss()
}

// flatten adds a new flatten layer to the network.
pub fn (mut nn Sequential<T>) flatten() {
	nn.info.flatten()
}

// relu adds a new relu layer to the network.
pub fn (mut nn Sequential<T>) relu() {
	nn.info.relu()
}

// leaky_relu adds a new leaky_relu layer to the network.
pub fn (mut nn Sequential<T>) leaky_relu() {
	nn.info.leaky_relu()
}

// elu adds a new elu layer to the network.
pub fn (mut nn Sequential<T>) elu() {
	nn.info.elu()
}

// sigmod adds a new sigmod layer to the network.
pub fn (mut nn Sequential<T>) sigmod() {
	nn.info.sigmod()
}

// sgd adds a new sgd optimizer to the network.
pub fn (mut nn Sequential<T>) sgd(config optimizers.SgdOptimizerConfig) {
	nn.info.sgd(config)
}

// adam adds a new adam optimizer to the network.
pub fn (mut nn Sequential<T>) adam(config optimizers.AdamOptimizerConfig) {
	nn.info.adam(config)
}

pub fn (mut nn Sequential<T>) forward(mut train autograd.Variable<T>) ?&autograd.Variable<T> {
	for layer in nn.info.layers {
		train = layers.layer_forward<T>(layer, mut train)?
	}
	return train
}

pub fn (mut nn Sequential<T>) loss(output &autograd.Variable<T>, target &vtl.Tensor<T>) ?&autograd.Variable<T> {
	return loss.loss_loss<T>(nn.info.loss, output, target)
}

pub fn (mut nn Sequential<T>) optimize() ? {
	return optimizers.optimize<T>(mut nn.info.optimizer)
}
