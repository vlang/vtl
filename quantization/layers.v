module quantization

import vtl

// QuantizedLinearLayer is a forward-only INT8 Linear inference layer.
@[heap]
pub struct QuantizedLinearLayer {
pub:
	weights       &vtl.Tensor[i8]
	weight_params PerChannelParams
	input_params  QuantizationParams
	bias          &vtl.Tensor[f32]
}

// quantize_linear_layer_f32 converts trained f32 weights and bias for INT8 inference.
pub fn quantize_linear_layer_f32(weights &vtl.Tensor[f32], bias &vtl.Tensor[f32], input_params QuantizationParams, qmin i8, qmax i8) !&QuantizedLinearLayer {
	if weights.rank() != 2 || weights.shape[0] == 0 {
		return error('quantized Linear weights must have shape [out_features, in_features] with non-zero out_features')
	}
	out_features := weights.shape[0]
	bias_is_vector := bias.rank() == 1 && bias.shape[0] == out_features
	bias_is_row := bias.rank() == 2 && bias.shape[0] == 1 && bias.shape[1] == out_features
	if !bias_is_vector && !bias_is_row {
		return error('quantized Linear bias must have shape [out_features] or [1, out_features]')
	}
	validate_params(input_params)!
	mut observer := new_per_channel_min_max_observer(0)
	observer.observe_f32(weights)!
	weight_params := observer.params(qmin, qmax)!
	quantized_weights := quantize_per_channel_f32(weights, weight_params)!
	return &QuantizedLinearLayer{
		weights:       quantized_weights
		weight_params: weight_params
		input_params:  input_params
		bias:          bias
	}
}

// forward runs the quantized Linear operation without autograd.
pub fn (layer &QuantizedLinearLayer) forward(input &vtl.Tensor[f32]) !&vtl.Tensor[f32] {
	return linear_int8_f32(input, layer.weights, layer.input_params, layer.weight_params,
		layer.bias)
}

// QuantizedConv2DLayer is a forward-only INT8 Conv2D inference layer.
pub struct QuantizedConv2DLayer {
pub:
	weights       &vtl.Tensor[i8]
	weight_params PerChannelParams
	input_params  QuantizationParams
	bias          &vtl.Tensor[f32]
	kernel_size   []int
	config        QuantizedConv2DConfig
}

// quantize_conv2d_layer_f32 converts trained f32 Conv2D weights and bias for INT8 inference.
pub fn quantize_conv2d_layer_f32(weights &vtl.Tensor[f32], bias &vtl.Tensor[f32], input_params QuantizationParams, kernel_size []int, config QuantizedConv2DConfig, qmin i8, qmax i8) !&QuantizedConv2DLayer {
	if weights.rank() != 4 || weights.shape[0] == 0 {
		return error('quantized Conv2D weights must have shape [out_channels, in_channels/groups, kernel_h, kernel_w]')
	}
	out_channels := weights.shape[0]
	if kernel_size.len != 2 || weights.shape[2] != kernel_size[0]
		|| weights.shape[3] != kernel_size[1] {
		return error('quantized Conv2D kernel_size must match the weight dimensions')
	}
	if bias.rank() != 2 || bias.shape[0] != 1 || bias.shape[1] != out_channels {
		return error('quantized Conv2D bias must have shape [1, out_channels]')
	}
	validate_params(input_params)!
	mut observer := new_per_channel_min_max_observer(0)
	observer.observe_f32(weights)!
	weight_params := observer.params(qmin, qmax)!
	quantized_weights := quantize_per_channel_f32(weights, weight_params)!
	return &QuantizedConv2DLayer{
		weights:       quantized_weights
		weight_params: weight_params
		input_params:  input_params
		bias:          bias
		kernel_size:   kernel_size.clone()
		config:        QuantizedConv2DConfig{
			padding:  config.padding.clone()
			stride:   config.stride.clone()
			dilation: config.dilation.clone()
			groups:   config.groups
		}
	}
}

// forward runs the quantized Conv2D operation without autograd.
pub fn (layer &QuantizedConv2DLayer) forward(input &vtl.Tensor[f32]) !&vtl.Tensor[f32] {
	return conv2d_int8_f32(input, layer.weights, layer.input_params, layer.weight_params,
		layer.bias, layer.kernel_size, layer.config)
}
