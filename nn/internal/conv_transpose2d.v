module internal

import vtl

@[params]
pub struct ConvTranspose2DConfig {
pub:
	stride         []int = [1, 1]
	padding        []int = [0, 0]
	output_padding []int = [0, 0]
	dilation       []int = [1, 1]
	groups         int   = 1
}

pub fn conv_transpose2d_forward[T](input &vtl.Tensor[T], weight &vtl.Tensor[T], bias &vtl.Tensor[T], config ConvTranspose2DConfig) !&vtl.Tensor[T] {
	output_shape := validate_conv_transpose2d(input, weight, bias, config)!
	batch, in_channels := input.shape[0], input.shape[1]
	out_channels := weight.shape[1] * config.groups
	in_per_group := in_channels / config.groups
	out_per_group := out_channels / config.groups
	mut output := vtl.zeros[T]([batch, out_channels, output_shape[0], output_shape[1]])
	for b in 0 .. batch {
		for ic in 0 .. in_channels {
			group := ic / in_per_group
			for ih in 0 .. input.shape[2] {
				for iw in 0 .. input.shape[3] {
					value := input.get([b, ic, ih, iw])
					for local_oc in 0 .. out_per_group {
						oc := group * out_per_group + local_oc
						for kh in 0 .. weight.shape[2] {
							oh := ih * config.stride[0] - config.padding[0] + kh * config.dilation[0]
							if oh < 0 || oh >= output_shape[0] {
								continue
							}
							for kw in 0 .. weight.shape[3] {
								ow := iw * config.stride[1] - config.padding[1] + kw * config.dilation[1]
								if ow >= 0 && ow < output_shape[1] {
									index := [b, oc, oh, ow]
									weight_index := [ic, local_oc, kh, kw]
									output.set(index, output.get(index) + value * weight.get(weight_index))
								}
							}
						}
					}
				}
			}
		}
	}
	for b in 0 .. batch {
		for oc in 0 .. out_channels {
			for oh in 0 .. output_shape[0] {
				for ow in 0 .. output_shape[1] {
					index := [b, oc, oh, ow]
					output.set(index, output.get(index) + bias.get_nth(oc))
				}
			}
		}
	}
	return output
}

pub fn conv_transpose2d_backward[T](grad_output &vtl.Tensor[T], input &vtl.Tensor[T], weight &vtl.Tensor[T], bias &vtl.Tensor[T], config ConvTranspose2DConfig) ![]&vtl.Tensor[T] {
	output_shape := validate_conv_transpose2d(input, weight, bias, config)!
	expected := [input.shape[0], weight.shape[1] * config.groups, output_shape[0], output_shape[1]]
	if grad_output.shape != expected {
		return error('conv_transpose2d_backward: grad_output shape must be ${expected}, got ${grad_output.shape}')
	}
	batch, in_channels := input.shape[0], input.shape[1]
	out_channels := weight.shape[1] * config.groups
	in_per_group := in_channels / config.groups
	out_per_group := out_channels / config.groups
	mut d_input := vtl.zeros_like[T](input)
	mut d_weight := vtl.zeros_like[T](weight)
	mut d_bias := vtl.zeros_like[T](bias)
	for b in 0 .. batch {
		for ic in 0 .. in_channels {
			group := ic / in_per_group
			for ih in 0 .. input.shape[2] {
				for iw in 0 .. input.shape[3] {
					for local_oc in 0 .. out_per_group {
						oc := group * out_per_group + local_oc
						for kh in 0 .. weight.shape[2] {
							oh := ih * config.stride[0] - config.padding[0] + kh * config.dilation[0]
							if oh < 0 || oh >= output_shape[0] {
								continue
							}
							for kw in 0 .. weight.shape[3] {
								ow := iw * config.stride[1] - config.padding[1] + kw * config.dilation[1]
								if ow >= 0 && ow < output_shape[1] {
									grad := grad_output.get([b, oc, oh, ow])
									weight_index := [ic, local_oc, kh, kw]
									d_input_index := [b, ic, ih, iw]
									d_input.set(d_input_index, d_input.get(d_input_index) + grad * weight.get(weight_index))
									d_weight.set(weight_index, d_weight.get(weight_index) + grad * input.get(d_input_index))
								}
							}
						}
					}
				}
			}
		}
	}
	for b in 0 .. batch {
		for oc in 0 .. out_channels {
			for oh in 0 .. output_shape[0] {
				for ow in 0 .. output_shape[1] {
					d_bias.set_nth(oc, d_bias.get_nth(oc) + grad_output.get([b, oc, oh, ow]))
				}
			}
		}
	}
	return [d_input, d_weight, d_bias]
}

fn validate_conv_transpose2d[T](input &vtl.Tensor[T], weight &vtl.Tensor[T], bias &vtl.Tensor[T], config ConvTranspose2DConfig) ![]int {
	if input.shape.len != 4 || weight.shape.len != 4 {
		return error('conv_transpose2d: input and weight must have rank 4')
	}
	if config.stride.len != 2 || config.padding.len != 2 || config.output_padding.len != 2
		|| config.dilation.len != 2 {
		return error('conv_transpose2d: stride, padding, output_padding, and dilation must have two values')
	}
	if config.groups <= 0 || config.stride[0] <= 0 || config.stride[1] <= 0
		|| config.dilation[0] <= 0 || config.dilation[1] <= 0 || config.padding[0] < 0
		|| config.padding[1] < 0 || config.output_padding[0] < 0 || config.output_padding[1] < 0 {
		return error('conv_transpose2d: invalid groups, stride, padding, output_padding, or dilation')
	}
	for axis in 0 .. 2 {
		if config.output_padding[axis] >= config.stride[axis]
			&& config.output_padding[axis] >= config.dilation[axis] {
			return error('conv_transpose2d: output_padding must be smaller than stride or dilation on each axis')
		}
	}
	in_channels := input.shape[1]
	if weight.shape[1] <= 0 || weight.shape[1] > max_int / config.groups {
		return error('conv_transpose2d: output channel count is invalid or too large')
	}
	out_channels := weight.shape[1] * config.groups
	if in_channels <= 0 || in_channels % config.groups != 0 || weight.shape[0] != in_channels
		|| bias.shape != [out_channels] {
		return error('conv_transpose2d: incompatible input, weight, bias, or group dimensions')
	}
	if input.shape[0] <= 0 || input.shape[2] <= 0 || input.shape[3] <= 0
		|| weight.shape[2] <= 0 || weight.shape[3] <= 0 {
		return error('conv_transpose2d: batch, input dimensions, and kernel dimensions must be positive')
	}
	mut output_shape := []int{len: 2}
	for axis in 0 .. 2 {
		input_extent := input.shape[axis + 2] - 1
		kernel_extent := weight.shape[axis + 2] - 1
		if input_extent > max_int / config.stride[axis]
			|| kernel_extent > max_int / config.dilation[axis]
			|| config.padding[axis] > max_int / 2 {
			return error('conv_transpose2d: dimensions exceed supported integer sizes')
		}
		input_span := input_extent * config.stride[axis]
		kernel_span := kernel_extent * config.dilation[axis]
		if input_span > max_int - kernel_span
			|| input_span + kernel_span > max_int - config.output_padding[axis] - 1 {
			return error('conv_transpose2d: dimensions exceed supported integer sizes')
		}
		output_shape[axis] = input_span + kernel_span + config.output_padding[axis] + 1 -
			2 * config.padding[axis]
		if output_shape[axis] <= 0 {
			return error('conv_transpose2d: calculated output dimensions must be positive')
		}
	}
	output_size := input.shape[0]
	for dim in [out_channels, output_shape[0], output_shape[1]] {
		if dim > 0 && output_size > max_int / dim {
			return error('conv_transpose2d: output tensor is too large')
		}
		output_size *= dim
	}
	return output_shape
}
