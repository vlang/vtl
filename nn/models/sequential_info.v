module models

import vtl.autograd
import vtl.nn.layers
import vtl.nn.loss
import vtl.nn.types

pub struct SequentialInfo[T] {
	ctx &autograd.Context[T] = unsafe { nil }
pub mut:
	layers []types.Layer[T]
	loss   types.Loss[T]
}

// sequential_info creates a new neural network container
// with an empty list of layers.
pub fn sequential_info[T](ctx &autograd.Context[T], layers_ []types.Layer[T]) &SequentialInfo[T] {
	return &SequentialInfo[T]{
		ctx:    ctx
		layers: layers_
		loss:   unsafe { nil }
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

// sigmoid adds a new sigmoid layer to the network.
pub fn (mut ls SequentialInfo[T]) sigmoid() {
	layer := ls.layers[ls.layers.len - 1]
	shape := layer.output_shape()
	ls.layers << layers.sigmoid_layer[T](ls.ctx, shape)
}

// tanh adds a new tanh layer to the network.
pub fn (mut ls SequentialInfo[T]) tanh() {
	layer := ls.layers[ls.layers.len - 1]
	shape := layer.output_shape()
	ls.layers << layers.tanh_layer[T](ls.ctx, shape)
}

// softmax adds a new softmax layer to the network.
pub fn (mut ls SequentialInfo[T]) softmax() {
	layer := ls.layers[ls.layers.len - 1]
	shape := layer.output_shape()
	ls.layers << layers.softmax_layer[T](ls.ctx, layers.SoftmaxLayerConfig{})
}

// gelu adds a new GELU layer to the network.
pub fn (mut ls SequentialInfo[T]) gelu() {
	layer := ls.layers[ls.layers.len - 1]
	shape := layer.output_shape()
	ls.layers << layers.gelu_layer[T](ls.ctx, shape)
}

// swish adds a new Swish layer to the network.
pub fn (mut ls SequentialInfo[T]) swish() {
	layer := ls.layers[ls.layers.len - 1]
	shape := layer.output_shape()
	ls.layers << layers.swish_layer[T](ls.ctx, shape)
}

// mish adds a new Mish layer to the network.
pub fn (mut ls SequentialInfo[T]) mish() {
	layer := ls.layers[ls.layers.len - 1]
	shape := layer.output_shape()
	ls.layers << layers.mish_layer[T](ls.ctx, shape)
}

// conv2d adds a new Conv2D layer to the network.
pub fn (mut ls SequentialInfo[T]) conv2d(in_channels int, out_channels int, kernel_size []int, config layers.Conv2DConfig) {
	prev_layer := ls.layers[ls.layers.len - 1]
	in_ch := if prev_layer.output_shape()[0] > 0 { prev_layer.output_shape()[0] } else { in_channels }
	ls.layers << layers.conv2d_layer[T](ls.ctx, in_ch, out_channels, kernel_size, config)
}

// batchnorm1d adds a new BatchNorm1D layer to the network.
pub fn (mut ls SequentialInfo[T]) batchnorm1d(num_features int, config layers.BatchNorm1DConfig) {
	ls.layers << layers.batchnorm1d_layer[T](ls.ctx, num_features, config)
}

// avgpool2d adds a new AveragePool2D layer to the network.
pub fn (mut ls SequentialInfo[T]) avgpool2d(kernel []int, padding []int, stride []int) {
	layer := ls.layers[ls.layers.len - 1]
	shape := layer.output_shape()
	ls.layers << layers.avgpool2d_layer[T](ls.ctx, shape, kernel, padding, stride)
}

// global_avgpool2d adds a new GlobalAveragePool2D layer to the network.
pub fn (mut ls SequentialInfo[T]) global_avgpool2d() {
	ls.layers << layers.global_avgpool2d_layer[T](ls.ctx)
}

// layer_norm adds a new LayerNorm layer to the network.
pub fn (mut ls SequentialInfo[T]) layer_norm(normalized_shape []int, config layers.LayerNormConfig) {
	ls.layers << layers.layer_norm_layer[T](ls.ctx, normalized_shape, config)
}

// embedding adds a new Embedding layer to the network.
pub fn (mut ls SequentialInfo[T]) embedding(vocab_size int, embedding_dim int) {
	ls.layers << layers.embedding_layer[T](ls.ctx, vocab_size, embedding_dim)
}

// lstm adds a new LSTM layer to the network.
pub fn (mut ls SequentialInfo[T]) lstm(input_size int, hidden_size int, num_layers int) {
	ls.layers << layers.lstm_layer[T](ls.ctx, input_size, hidden_size, num_layers)
}

// multihead_attention adds a new MultiHeadAttention layer to the network.
pub fn (mut ls SequentialInfo[T]) multihead_attention(embed_dim int, num_heads int) {
	ls.layers << layers.multihead_attention_layer[T](ls.ctx, embed_dim, num_heads)
}

// positional_encoding adds a new PositionalEncoding layer to the network.
pub fn (mut ls SequentialInfo[T]) positional_encoding(embed_dim int, max_len int) {
	ls.layers << layers.positional_encoding_layer[T](ls.ctx, embed_dim, max_len) or { panic(err) }
}

// cross_entropy_loss sets the loss function to cross entropy loss.
pub fn (mut ls SequentialInfo[T]) cross_entropy_loss() {
	ls.loss = loss.cross_entropy_loss[T]()
}

// bce_loss sets the loss function to binary cross entropy loss.
pub fn (mut ls SequentialInfo[T]) bce_loss() {
	ls.loss = loss.bce_loss[T](loss.BCELossConfig{})
}

// huber_loss sets the loss function to Huber loss.
pub fn (mut ls SequentialInfo[T]) huber_loss() {
	ls.loss = loss.huber_loss[T](loss.HuberLossConfig{})
}

// nll_loss sets the loss function to negative log likelihood loss.
pub fn (mut ls SequentialInfo[T]) nll_loss() {
	ls.loss = loss.nll_loss[T](unsafe { nil })
}

// kl_div_loss sets the loss function to KL divergence loss.
pub fn (mut ls SequentialInfo[T]) kl_div_loss() {
	ls.loss = loss.kl_div_loss[T]()
}
