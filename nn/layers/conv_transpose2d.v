module layers

import vtl
import vtl.autograd
import vtl.nn.gates.layers
import vtl.nn.internal
import vtl.nn.types

@[params]
pub struct ConvTranspose2DConfig {
pub:
	stride         []int = [1, 1]
	padding        []int = [0, 0]
	output_padding []int = [0, 0]
	dilation       []int = [1, 1]
	groups         int   = 1
}

pub struct ConvTranspose2DLayer[T] {
pub:
	in_channels  int
	out_channels int
	kernel_size  []int
	input_shape  []int
	config       ConvTranspose2DConfig
pub mut:
	weight &autograd.Variable[T] = unsafe { nil }
	bias   &autograd.Variable[T] = unsafe { nil }
}

pub fn conv_transpose2d_layer[T](ctx &autograd.Context[T], in_channels int, out_channels int, kernel_size []int, config ConvTranspose2DConfig, input_shape []int) types.Layer[T] {
	if in_channels <= 0 || out_channels <= 0 || kernel_size.len != 2 || kernel_size[0] <= 0
		|| kernel_size[1] <= 0 || config.groups <= 0 || in_channels % config.groups != 0
		|| out_channels % config.groups != 0 {
		panic('conv_transpose2d_layer: channels and kernel dimensions must be positive, and groups must divide channels')
	}
	if config.stride.len != 2 || config.padding.len != 2 || config.output_padding.len != 2
		|| config.dilation.len != 2 {
		panic('conv_transpose2d_layer: stride, padding, output_padding, and dilation must have two values')
	}
	for axis in 0 .. 2 {
		if config.stride[axis] <= 0 || config.dilation[axis] <= 0 || config.padding[axis] < 0
			|| config.output_padding[axis] < 0
			|| (config.output_padding[axis] >= config.stride[axis]
				&& config.output_padding[axis] >= config.dilation[axis]) {
			panic('conv_transpose2d_layer: invalid stride, padding, output_padding, or dilation')
		}
	}
	weight := internal.variance_scaled[T]([in_channels, out_channels / config.groups, kernel_size[0],
		kernel_size[1]], vtl.cast[T](2), .fan_out, .normal)
	bias := vtl.zeros[T]([out_channels])
	layer := &ConvTranspose2DLayer[T]{
		in_channels:  in_channels
		out_channels: out_channels
		kernel_size:  kernel_size.clone()
		input_shape:  input_shape.clone()
		config:       config
		weight:       ctx.variable(weight)
		bias:         ctx.variable(bias)
	}
	return types.layer[T](voidptr(layer), conv_transpose2_d_layer_output_shape_dispatch[T],
		conv_transpose2_d_layer_variables_dispatch[T], conv_transpose2_d_layer_forward_dispatch[T])
}

pub fn (layer &ConvTranspose2DLayer[T]) output_shape() []int {
	if layer.input_shape.len != 3 || layer.input_shape[1] <= 0 || layer.input_shape[2] <= 0 {
		return [layer.out_channels, -1, -1]
	}
	mut shape := [layer.out_channels]
	for axis in 0 .. 2 {
		shape << (layer.input_shape[axis + 1] - 1) * layer.config.stride[axis] -
			2 * layer.config.padding[axis] + layer.config.dilation[axis] * (layer.kernel_size[axis] - 1) +
			layer.config.output_padding[axis] + 1
	}
	return shape
}

pub fn (layer &ConvTranspose2DLayer[T]) variables() []&autograd.Variable[T] {
	return [layer.weight, layer.bias]
}

pub fn (layer &ConvTranspose2DLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	if input.context != layer.weight.context {
		return error('ConvTranspose2DLayer.forward: input and layer must share an autograd context')
	}
	if input.value.shape.len != 4 || input.value.shape[1] != layer.in_channels {
		return error('ConvTranspose2DLayer.forward: input must have shape [batch, ${layer.in_channels}, height, width]')
	}
	config := internal.ConvTranspose2DConfig{
		stride:         layer.config.stride
		padding:        layer.config.padding
		output_padding: layer.config.output_padding
		dilation:       layer.config.dilation
		groups:         layer.config.groups
	}
	output := internal.conv_transpose2d_forward[T](input.value, layer.weight.value, layer.bias.value,
		config)!
	mut result := input.context.variable(output)
	if input.is_grad_needed() || layer.weight.is_grad_needed() || layer.bias.is_grad_needed() {
		gate := layers.conv_transpose2d_gate[T](input, layer.weight, layer.bias, config)
		gate.cache(mut result, input, layer.weight, layer.bias)!
	}
	return result
}

fn conv_transpose2_d_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&ConvTranspose2DLayer[T](layer)).output_shape() }
}

fn conv_transpose2_d_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&ConvTranspose2DLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn conv_transpose2_d_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&ConvTranspose2DLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}
