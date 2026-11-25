module layers

import vtl
import vtl.autograd
import vtl.nn.gates.layers
import vtl.nn.internal
import vtl.nn.types

@[params]
pub struct ConvTranspose1DConfig {
pub:
	stride         int = 1
	padding        int
	output_padding int
	dilation       int = 1
	groups         int = 1
}

pub struct ConvTranspose1DLayer[T] {
pub:
	in_channels  int
	out_channels int
	kernel_size  int
	input_length int
	config       ConvTranspose1DConfig
pub mut:
	weight &autograd.Variable[T] = unsafe { nil }
	bias   &autograd.Variable[T] = unsafe { nil }
}

pub fn conv_transpose1d_layer[T](ctx &autograd.Context[T], in_channels int, out_channels int, kernel_size int, config ConvTranspose1DConfig, input_length int) types.Layer[T] {
	if in_channels <= 0 || out_channels <= 0 || kernel_size <= 0 || config.groups <= 0
		|| in_channels % config.groups != 0 || out_channels % config.groups != 0 {
		panic('conv_transpose1d_layer: channels, kernel_size, and groups must be positive and channels divisible by groups')
	}
	// ConvTranspose stores [in, out/groups, kernel], so fan_out gives its fan-in.
	weight := internal.variance_scaled[T]([in_channels, out_channels / config.groups, kernel_size], vtl.cast[T](2), .fan_out, .normal)
	bias := vtl.zeros[T]([out_channels])
	layer := &ConvTranspose1DLayer[T]{
		in_channels:  in_channels
		out_channels: out_channels
		kernel_size:  kernel_size
		input_length: input_length
		config:       config
		weight:       ctx.variable(weight)
		bias:         ctx.variable(bias)
	}
	return types.layer[T](voidptr(layer), conv_transpose1d_layer_output_shape_dispatch[T],
		conv_transpose1d_layer_variables_dispatch[T], conv_transpose1d_layer_forward_dispatch[T])
}

pub fn (layer &ConvTranspose1DLayer[T]) output_shape() []int {
	if layer.input_length < 0 || layer.config.stride <= 0 || layer.config.dilation <= 0 {
		return [layer.out_channels, -1]
	}
	length := (layer.input_length - 1) * layer.config.stride - 2 * layer.config.padding +
		layer.config.dilation * (layer.kernel_size - 1) + layer.config.output_padding + 1
	return [layer.out_channels, length]
}

pub fn (layer &ConvTranspose1DLayer[T]) variables() []&autograd.Variable[T] {
	return [layer.weight, layer.bias]
}

pub fn (layer &ConvTranspose1DLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	if input.context != layer.weight.context {
		return error('ConvTranspose1DLayer.forward: input and layer must share an autograd context')
	}
	if input.value.shape.len != 3 || input.value.shape[1] != layer.in_channels {
		return error('ConvTranspose1DLayer.forward: input must have shape [batch, ${layer.in_channels}, length]')
	}
	config := internal.ConvTranspose1DConfig{
		stride:         layer.config.stride
		padding:        layer.config.padding
		output_padding: layer.config.output_padding
		dilation:       layer.config.dilation
		groups:         layer.config.groups
	}
	output := internal.conv_transpose1d_forward[T](input.value, layer.weight.value, layer.bias.value, config)!
	mut result := input.context.variable(output)
	if input.is_grad_needed() || layer.weight.is_grad_needed() || layer.bias.is_grad_needed() {
		gate := layers.conv_transpose1d_gate[T](input, layer.weight, layer.bias, config)
		gate.cache(mut result, input, layer.weight, layer.bias)!
	}
	return result
}

fn conv_transpose1d_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&ConvTranspose1DLayer[T](layer)).output_shape() }
}

fn conv_transpose1d_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&ConvTranspose1DLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn conv_transpose1d_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&ConvTranspose1DLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}
