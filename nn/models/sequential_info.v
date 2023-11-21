module models

import vtl.autograd
import vtl.nn.layers
import vtl.nn.loss
import vtl.nn.types

pub struct SequentialInfo[T] {
	ctx &autograd.Context[T] = unsafe { nil }
pub mut:
	layers []types.Layer[T]
	loss   types.Loss
}

// sequential_info creates a new neural network container
// with an empty list of layers.
pub fn sequential_info[T](ctx &autograd.Context[T], layers_ []types.Layer[T]) &SequentialInfo[T] {
	return &SequentialInfo[T]{
		ctx: ctx
		layers: layers_
		loss: unsafe { nil }
	}
}

// input adds a new input layer to the network
// with the given shape.
pub fn (mut ls SequentialInfo[T]) input(shape []int) {
	ls.layers << layers.input_layer[T](ls.ctx, shape)
}

// linear adds a new linear layer to the network
// with the given output size
pub fn (mut ls SequentialInfo[T]) linear(output_size int) {
	layer := ls.layers[ls.layers.len - 1]
	input_size := layer.output_shape()[0]
	ls.layers << layers.linear_layer[T](ls.ctx, input_size, output_size)
}

// maxpool2d adds a new maxpool2d layer to the network
// with the given kernel size and stride.
pub fn (mut ls SequentialInfo[T]) maxpool2d(kernel []int, padding []int, stride []int) {
	layer := ls.layers[ls.layers.len - 1]
	shape := layer.output_shape()
	ls.layers << layers.maxpool2d_layer[T](ls.ctx, shape, kernel, padding, stride)
}

// mse_loss sets the loss function to the mean squared error loss.
pub fn (mut ls SequentialInfo[T]) mse_loss() {
	ls.loss = loss.mse_loss[T]()
}

// sigmoid_cross_entropy_loss sets the loss function to the sigmoid cross entropy loss.
pub fn (mut ls SequentialInfo[T]) sigmoid_cross_entropy_loss() {
	ls.loss = loss.sigmoid_cross_entropy_loss[T]()
}

// softmax_cross_entropy_loss sets the loss function to the softmax cross entropy loss.
pub fn (mut ls SequentialInfo[T]) softmax_cross_entropy_loss() {
	ls.loss = loss.softmax_cross_entropy_loss[T]()
}

// flatten adds a new flatten layer to the network.
pub fn (mut ls SequentialInfo[T]) flatten() {
	layer := ls.layers[ls.layers.len - 1]
	shape := layer.output_shape()
	ls.layers << layers.flatten_layer[T](ls.ctx, shape)
}

// relu adds a new relu layer to the network.
pub fn (mut ls SequentialInfo[T]) relu() {
	layer := ls.layers[ls.layers.len - 1]
	shape := layer.output_shape()
	ls.layers << layers.relu_layer[T](ls.ctx, shape)
}

// leaky_relu adds a new leaky_relu layer to the network.
pub fn (mut ls SequentialInfo[T]) leaky_relu() {
	layer := ls.layers[ls.layers.len - 1]
	shape := layer.output_shape()
	ls.layers << layers.leaky_relu_layer[T](ls.ctx, shape)
}

// elu adds a new elu layer to the network.
pub fn (mut ls SequentialInfo[T]) elu() {
	layer := ls.layers[ls.layers.len - 1]
	shape := layer.output_shape()
	ls.layers << layers.elu_layer[T](ls.ctx, shape)
}

// sigmod adds a new sigmod layer to the network.
pub fn (mut ls SequentialInfo[T]) sigmod() {
	layer := ls.layers[ls.layers.len - 1]
	shape := layer.output_shape()
	ls.layers << layers.sigmoid_layer[T](ls.ctx, shape)
}
