module models

import vtl
import vtl.autograd
import vtl.nn.loss
import vtl.nn.types
import vtl.nn.layers

fn init() {
	println(@MOD + ' module is a WIP and not yet functional')
}

pub struct Sequential[T] {
pub mut:
	info &SequentialInfo[T] = unsafe { nil }
}

// sequential creates a new sequential network with a new context.
pub fn sequential[T]() &Sequential[T] {
	ctx := autograd.ctx[T]()
	empty_layers := []types.Layer[T]{}
	return &Sequential[T]{
		info: sequential_info[T](ctx, empty_layers)
	}
}

// sequential_with_layers creates a new sequential network with a new context
// and the given layers.
pub fn sequential_with_layers[T](given_layers []types.Layer[T]) &Sequential[T] {
	ctx := autograd.ctx[T]()
	return &Sequential[T]{
		info: sequential_info[T](ctx, given_layers)
	}
}

// sequential_from_ctx creates a new sequential network with the given context.
pub fn sequential_from_ctx[T](ctx &autograd.Context[T]) &Sequential[T] {
	empty_layers := []types.Layer[T]{}
	return &Sequential[T]{
		info: sequential_info[T](ctx, empty_layers)
	}
}

// sequential_from_ctx_with_layers creates a new sequential network with the given context
// and the given layers.
pub fn sequential_from_ctx_with_layers[T](ctx &autograd.Context[T], given_layers []types.Layer[T]) &Sequential[T] {
	return &Sequential[T]{
		info: sequential_info[T](ctx, given_layers)
	}
}

// input adds a new input layer to the network
// with the given shape.
pub fn (mut nn Sequential[T]) input(shape []int) {
	nn.info.input(shape)
}

// linear adds a new linear layer to the network
// with the given output size
pub fn (mut nn Sequential[T]) linear(output_size int) {
	nn.info.linear(output_size)
}

// maxpool2d adds a new maxpool2d layer to the network
// with the given kernel size and stride.
pub fn (mut nn Sequential[T]) maxpool2d(kernel []int, padding []int, stride []int) {
	nn.info.maxpool2d(kernel, padding, stride)
}

// mse_loss sets the loss function to the mean squared error loss.
pub fn (mut nn Sequential[T]) mse_loss() {
	nn.info.mse_loss()
}

// sigmoid_cross_entropy_loss sets the loss function to the sigmoid cross entropy loss.
pub fn (mut nn Sequential[T]) sigmoid_cross_entropy_loss() {
	nn.info.sigmoid_cross_entropy_loss()
}

// softmax_cross_entropy_loss sets the loss function to the softmax cross entropy loss.
pub fn (mut nn Sequential[T]) softmax_cross_entropy_loss() {
	nn.info.softmax_cross_entropy_loss()
}

// flatten adds a new flatten layer to the network.
pub fn (mut nn Sequential[T]) flatten() {
	nn.info.flatten()
}

// relu adds a new relu layer to the network.
pub fn (mut nn Sequential[T]) relu() {
	nn.info.relu()
}

// leaky_relu adds a new leaky_relu layer to the network.
pub fn (mut nn Sequential[T]) leaky_relu() {
	nn.info.leaky_relu()
}

// elu adds a new elu layer to the network.
pub fn (mut nn Sequential[T]) elu() {
	nn.info.elu()
}

// sigmoid adds a new sigmoid layer to the network.
pub fn (mut nn Sequential[T]) sigmoid() {
	nn.info.sigmoid()
}

// tanh adds a new tanh layer to the network.
pub fn (mut nn Sequential[T]) tanh() {
	nn.info.tanh()
}

// softmax adds a new softmax layer to the network.
pub fn (mut nn Sequential[T]) softmax() {
	nn.info.softmax()
}

// gelu adds a new GELU layer to the network.
pub fn (mut nn Sequential[T]) gelu() {
	nn.info.gelu()
}

// swish adds a new Swish layer to the network.
pub fn (mut nn Sequential[T]) swish() {
	nn.info.swish()
}

// mish adds a new Mish layer to the network.
pub fn (mut nn Sequential[T]) mish() {
	nn.info.mish()
}

// conv2d adds a new Conv2D layer to the network.
pub fn (mut nn Sequential[T]) conv2d(in_channels int, out_channels int, kernel_size []int, config layers.Conv2DConfig) {
	nn.info.conv2d(in_channels, out_channels, kernel_size, config)
}

// batchnorm1d adds a new BatchNorm1D layer to the network.
pub fn (mut nn Sequential[T]) batchnorm1d(num_features int, config layers.BatchNorm1DConfig) {
	nn.info.batchnorm1d(num_features, config)
}

// avgpool2d adds a new AveragePool2D layer to the network.
pub fn (mut nn Sequential[T]) avgpool2d(kernel []int, padding []int, stride []int) {
	nn.info.avgpool2d(kernel, padding, stride)
}

// global_avgpool2d adds a new GlobalAveragePool2D layer to the network.
pub fn (mut nn Sequential[T]) global_avgpool2d() {
	nn.info.global_avgpool2d()
}

// layer_norm adds a new LayerNorm layer to the network.
pub fn (mut nn Sequential[T]) layer_norm(normalized_shape []int, config layers.LayerNormConfig) {
	nn.info.layer_norm(normalized_shape, config)
}

// embedding adds a new Embedding layer to the network.
pub fn (mut nn Sequential[T]) embedding(vocab_size int, embedding_dim int) {
	nn.info.embedding(vocab_size, embedding_dim)
}

// lstm adds a new LSTM layer to the network.
pub fn (mut nn Sequential[T]) lstm(input_size int, hidden_size int, num_layers int) {
	nn.info.lstm(input_size, hidden_size, num_layers)
}

// multihead_attention adds a new MultiHeadAttention layer to the network.
pub fn (mut nn Sequential[T]) multihead_attention(embed_dim int, num_heads int) {
	nn.info.multihead_attention(embed_dim, num_heads)
}

// positional_encoding adds a new PositionalEncoding layer to the network.
pub fn (mut nn Sequential[T]) positional_encoding(embed_dim int, max_len int) {
	nn.info.positional_encoding(embed_dim, max_len)
}

// cross_entropy_loss sets the loss function to cross entropy loss.
pub fn (mut nn Sequential[T]) cross_entropy_loss() {
	nn.info.cross_entropy_loss()
}

// bce_loss sets the loss function to binary cross entropy loss.
pub fn (mut nn Sequential[T]) bce_loss() {
	nn.info.bce_loss()
}

// huber_loss sets the loss function to Huber loss.
pub fn (mut nn Sequential[T]) huber_loss() {
	nn.info.huber_loss()
}

// nll_loss sets the loss function to negative log likelihood loss.
pub fn (mut nn Sequential[T]) nll_loss() {
	nn.info.nll_loss()
}

// kl_div_loss sets the loss function to KL divergence loss.
pub fn (mut nn Sequential[T]) kl_div_loss() {
	nn.info.kl_div_loss()
}

pub fn (mut nn Sequential[T]) forward(train &autograd.Variable[T]) !&autograd.Variable[T] {
	mut cur := &autograd.Variable[T]{
		context: train.context
		value: train.value
		grad: train.grad
		requires_grad: train.requires_grad
	}
	for layer in nn.info.layers {
		cur = layer.forward(cur)!
	}
	return cur
}

pub fn (mut nn Sequential[T]) loss(output &autograd.Variable[T], target &vtl.Tensor[T]) !&autograd.Variable[T] {
	return nn.info.loss.loss(output, target)
}
