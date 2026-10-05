module layers

import vtl
import vtl.autograd
import vtl.nn.gates.layers
import vtl.nn.internal
import vtl.nn.types

// Conv1DConfig configures stride, padding, dilation, and channel groups.
@[params]
pub struct Conv1DConfig {
pub:
	stride   int = 1
	padding  int
	dilation int = 1
	groups   int = 1
}

// Conv1DLayer applies grouped cross-correlation to [batch, channels, length].
pub struct Conv1DLayer[T] {
pub:
	in_channels  int
	out_channels int
	kernel_size  int
	input_length int
	config       Conv1DConfig
pub mut:
	weight &autograd.Variable[T] = unsafe { nil }
	bias   &autograd.Variable[T] = unsafe { nil }
}

// conv1d_layer creates a 1D convolution layer. Pass input_length as -1 when the
// input length is dynamic; output_shape then reports -1 for its length.
pub fn conv1d_layer[T](ctx &autograd.Context[T], in_channels int, out_channels int,
	kernel_size int, config Conv1DConfig, input_length int) types.Layer[T] {
	if in_channels <= 0 || out_channels <= 0 || kernel_size <= 0 || config.groups <= 0
		|| in_channels % config.groups != 0 || out_channels % config.groups != 0 {
		panic('conv1d_layer: channels, kernel_size, and groups must be positive and channels divisible by groups')
	}
	weight := internal.kaiming_normal[T]([out_channels, in_channels / config.groups, kernel_size])
	bias := vtl.zeros[T]([out_channels])
	layer := &Conv1DLayer[T]{
		in_channels:  in_channels
		out_channels: out_channels
		kernel_size:  kernel_size
		input_length: input_length
		config:       config
		weight:       ctx.variable(weight)
		bias:         ctx.variable(bias)
	}
	return types.layer[T](voidptr(layer), conv1d_layer_output_shape_dispatch[T],
		conv1d_layer_variables_dispatch[T], conv1d_layer_forward_dispatch[T])
}

// output_shape returns [out_channels, output_length] excluding batch.
pub fn (l &Conv1DLayer[T]) output_shape() []int {
	if l.input_length < 0 || l.config.stride <= 0 || l.config.dilation <= 0 {
		return [l.out_channels, -1]
	}
	effective_kernel := l.config.dilation * (l.kernel_size - 1) + 1
	length := (l.input_length + 2 * l.config.padding - effective_kernel) / l.config.stride + 1
	return [l.out_channels, length]
}

// variables returns the trainable weight and bias variables.
pub fn (l &Conv1DLayer[T]) variables() []&autograd.Variable[T] {
	return [l.weight, l.bias]
}

// forward computes Conv1D and registers its reverse-mode derivative.
pub fn (l &Conv1DLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	if input.context != l.weight.context {
		return error('Conv1DLayer.forward: input and layer must share an autograd context')
	}
	if input.value.shape.len != 3 || input.value.shape[1] != l.in_channels {
		return error('Conv1DLayer.forward: input must have shape [batch, ${l.in_channels}, length]')
	}
	cfg := internal.Conv1DConfig{
		stride:   l.config.stride
		padding:  l.config.padding
		dilation: l.config.dilation
		groups:   l.config.groups
	}
	output := internal.conv1d_forward[T](input.value, l.weight.value, l.bias.value, cfg)!
	mut result := input.context.variable(output)
	if input.is_grad_needed() || l.weight.is_grad_needed() || l.bias.is_grad_needed() {
		gate := layers.conv1d_gate[T](input, l.weight, l.bias, cfg)
		gate.cache(mut result, input, l.weight, l.bias)!
	}
	return result
}

fn conv1d_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&Conv1DLayer[T](layer)).output_shape() }
}

fn conv1d_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&Conv1DLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn conv1d_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&Conv1DLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}
