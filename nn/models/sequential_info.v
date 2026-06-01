module models

import vtl.autograd
import vtl.nn.layers
import vtl.nn.loss
import vtl.nn.types

// SequentialInfo defines a public data structure for this module.
pub struct SequentialInfo[T] {
	ctx &autograd.Context[T] = unsafe { nil }
pub mut:
	layers        []types.Layer[T]
	layer_types   []string
	layer_configs []map[string]int
	loss          types.Loss[T]
}

// sequential_info creates a new neural network container
// with an empty list of layers.
pub fn sequential_info[T](ctx &autograd.Context[T], layers_ []types.Layer[T]) &SequentialInfo[T] {
	mut layer_types := []string{cap: layers_.len}
	mut layer_configs := []map[string]int{cap: layers_.len}
	for _ in layers_ {
		layer_types << ''
		layer_configs << map[string]int{}
	}
	return &SequentialInfo[T]{
		ctx:           ctx
		layers:        layers_
		layer_types:   layer_types
		layer_configs: layer_configs
	}
}

fn (mut ls SequentialInfo[T]) add_layer(layer types.Layer[T], layer_type string, config map[string]int) {
	ls.layers << layer
	ls.layer_types << layer_type
	ls.layer_configs << config.clone()
}

// input adds a new input layer to the network
// with the given shape.
pub fn (mut ls SequentialInfo[T]) input(shape []int) {
	mut config := map[string]int{}
	for i, dim in shape {
		config['shape_${i}'] = dim
	}
	ls.add_layer(layers.input_layer[T](ls.ctx, shape), 'InputLayer', config)
}

// linear adds a new linear layer to the network
// with the given output size
pub fn (mut ls SequentialInfo[T]) linear(output_size int) {
	layer := ls.layers[ls.layers.len - 1]
	input_size := layer.output_shape()[0]
	ls.add_layer(layers.linear_layer[T](ls.ctx, input_size, output_size), 'LinearLayer', {
		'in_features':  input_size
		'out_features': output_size
	})
}

// maxpool2d adds a new maxpool2d layer to the network
// with the given kernel size and stride.
pub fn (mut ls SequentialInfo[T]) maxpool2d(kernel []int, padding []int, stride []int) {
	layer := ls.layers[ls.layers.len - 1]
	shape := layer.output_shape()
	ls.add_layer(layers.maxpool2d_layer[T](ls.ctx, shape, kernel, padding, stride),
		'MaxPool2DLayer', {
		'kernel_h':  kernel[0]
		'kernel_w':  kernel[1]
		'padding_h': padding[0]
		'padding_w': padding[1]
		'stride_h':  stride[0]
		'stride_w':  stride[1]
	})
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
	ls.add_layer(layers.flatten_layer[T](ls.ctx, shape), 'FlattenLayer', {})
}

// relu adds a new relu layer to the network.
pub fn (mut ls SequentialInfo[T]) relu() {
	layer := ls.layers[ls.layers.len - 1]
	shape := layer.output_shape()
	ls.add_layer(layers.relu_layer[T](ls.ctx, shape), 'ReLULayer', {})
}

// leaky_relu adds a new leaky_relu layer to the network.
pub fn (mut ls SequentialInfo[T]) leaky_relu() {
	layer := ls.layers[ls.layers.len - 1]
	shape := layer.output_shape()
	ls.add_layer(layers.leaky_relu_layer[T](ls.ctx, shape), 'LeakyReLULayer', {})
}

// elu adds a new elu layer to the network.
pub fn (mut ls SequentialInfo[T]) elu() {
	layer := ls.layers[ls.layers.len - 1]
	shape := layer.output_shape()
	ls.add_layer(layers.elu_layer[T](ls.ctx, shape), 'ELULayer', {})
}

// sigmoid adds a new sigmoid layer to the network.
pub fn (mut ls SequentialInfo[T]) sigmoid() {
	layer := ls.layers[ls.layers.len - 1]
	shape := layer.output_shape()
	ls.add_layer(layers.sigmoid_layer[T](ls.ctx, shape), 'SigmoidLayer', {})
}

// tanh adds a new tanh layer to the network.
pub fn (mut ls SequentialInfo[T]) tanh() {
	layer := ls.layers[ls.layers.len - 1]
	shape := layer.output_shape()
	ls.add_layer(layers.tanh_layer[T](ls.ctx, shape), 'TanhLayer', {})
}

// softmax adds a new softmax layer to the network.
pub fn (mut ls SequentialInfo[T]) softmax() {
	ls.add_layer(layers.softmax_layer[T](ls.ctx, layers.SoftmaxLayerConfig{}), 'SoftmaxLayer', {})
}

// gelu adds a new GELU layer to the network.
pub fn (mut ls SequentialInfo[T]) gelu() {
	layer := ls.layers[ls.layers.len - 1]
	shape := layer.output_shape()
	ls.add_layer(layers.gelu_layer[T](ls.ctx, shape), 'GELULayer', {})
}

// swish adds a new Swish layer to the network.
pub fn (mut ls SequentialInfo[T]) swish() {
	layer := ls.layers[ls.layers.len - 1]
	shape := layer.output_shape()
	ls.add_layer(layers.swish_layer[T](ls.ctx, shape), 'SwishLayer', {})
}

// mish adds a new Mish layer to the network.
pub fn (mut ls SequentialInfo[T]) mish() {
	layer := ls.layers[ls.layers.len - 1]
	shape := layer.output_shape()
	ls.add_layer(layers.mish_layer[T](ls.ctx, shape), 'MishLayer', {})
}

// conv2d adds a new Conv2D layer to the network.
pub fn (mut ls SequentialInfo[T]) conv2d(in_channels int, out_channels int, kernel_size []int, config layers.Conv2DConfig) {
	prev_layer := ls.layers[ls.layers.len - 1]
	in_ch := if prev_layer.output_shape()[0] > 0 {
		prev_layer.output_shape()[0]
	} else {
		in_channels
	}
	ls.add_layer(layers.conv2d_layer[T](ls.ctx, in_ch, out_channels, kernel_size, config,
		prev_layer.output_shape()), 'Conv2DLayer', {
		'in_channels':  in_ch
		'out_channels': out_channels
		'kernel_h':     kernel_size[0]
		'kernel_w':     kernel_size[1]
		'padding_h':    config.padding[0]
		'padding_w':    config.padding[1]
		'stride_h':     config.stride[0]
		'stride_w':     config.stride[1]
		'dilation_h':   config.dilation[0]
		'dilation_w':   config.dilation[1]
		'groups':       config.groups
	})
}

// batchnorm1d adds a new BatchNorm1D layer to the network.
pub fn (mut ls SequentialInfo[T]) batchnorm1d(num_features int, config layers.BatchNorm1DConfig) {
	ls.add_layer(layers.batchnorm1d_layer[T](ls.ctx, num_features, config), 'BatchNorm1DLayer', {
		'num_features': num_features
	})
}

// avgpool2d adds a new AveragePool2D layer to the network.
pub fn (mut ls SequentialInfo[T]) avgpool2d(kernel []int, padding []int, stride []int) {
	layer := ls.layers[ls.layers.len - 1]
	shape := layer.output_shape()
	ls.add_layer(layers.avgpool2d_layer[T](ls.ctx, shape, kernel, padding, stride),
		'AvgPool2DLayer', {
		'kernel_h':  kernel[0]
		'kernel_w':  kernel[1]
		'padding_h': padding[0]
		'padding_w': padding[1]
		'stride_h':  stride[0]
		'stride_w':  stride[1]
	})
}

// global_avgpool2d adds a new GlobalAveragePool2D layer to the network.
pub fn (mut ls SequentialInfo[T]) global_avgpool2d() {
	ls.add_layer(layers.global_avgpool2d_layer[T](ls.ctx), 'GlobalAvgPool2DLayer', {})
}

// layer_norm adds a new LayerNorm layer to the network.
pub fn (mut ls SequentialInfo[T]) layer_norm(normalized_shape []int, config layers.LayerNormConfig) {
	mut layer_config := map[string]int{}
	for i, dim in normalized_shape {
		layer_config['normalized_shape_${i}'] = dim
	}
	ls.add_layer(layers.layer_norm_layer[T](ls.ctx, normalized_shape, config), 'LayerNormLayer',
		layer_config)
}

// embedding adds a new Embedding layer to the network.
pub fn (mut ls SequentialInfo[T]) embedding(vocab_size int, embedding_dim int) {
	ls.add_layer(layers.embedding_layer[T](ls.ctx, vocab_size, embedding_dim), 'EmbeddingLayer', {
		'vocab_size':    vocab_size
		'embedding_dim': embedding_dim
	})
}

// lstm adds a new LSTM layer to the network.
pub fn (mut ls SequentialInfo[T]) lstm(input_size int, hidden_size int, num_layers int) {
	ls.add_layer(layers.lstm_layer[T](ls.ctx, input_size, hidden_size, num_layers), 'LSTMLayer', {
		'input_size':  input_size
		'hidden_size': hidden_size
		'num_layers':  num_layers
	})
}

// multihead_attention adds a new MultiHeadAttention layer to the network.
pub fn (mut ls SequentialInfo[T]) multihead_attention(embed_dim int, num_heads int) {
	ls.add_layer(layers.multihead_attention_layer[T](ls.ctx, embed_dim, num_heads),
		'MultiHeadAttentionLayer', {
		'embed_dim': embed_dim
		'num_heads': num_heads
		'head_dim':  embed_dim / num_heads
	})
}

// positional_encoding adds a new PositionalEncoding layer to the network.
pub fn (mut ls SequentialInfo[T]) positional_encoding(embed_dim int, max_len int) {
	ls.add_layer(layers.positional_encoding_layer[T](ls.ctx, embed_dim, max_len) or { panic(err) },
		'PositionalEncodingLayer', {
		'embed_dim': embed_dim
		'max_len':   max_len
	})
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
