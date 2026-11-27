module internal

import vtl

@[params]
pub struct ConvTranspose1DConfig {
pub:
	stride         int = 1
	padding        int
	output_padding int
	dilation       int = 1
	groups         int = 1
}

pub fn conv_transpose1d_forward[T](input &vtl.Tensor[T], weight &vtl.Tensor[T], bias &vtl.Tensor[T], config ConvTranspose1DConfig) !&vtl.Tensor[T] {
	output_length := validate_conv_transpose1d(input, weight, bias, config)!
	batch, in_channels := input.shape[0], input.shape[1]
	out_channels := weight.shape[1] * config.groups
	in_per_group := in_channels / config.groups
	out_per_group := out_channels / config.groups
	mut output := vtl.zeros[T]([batch, out_channels, output_length])
	for b in 0 .. batch {
		for ic in 0 .. in_channels {
			group := ic / in_per_group
			for input_pos in 0 .. input.shape[2] {
				value := input.get([b, ic, input_pos])
				for local_oc in 0 .. out_per_group {
					oc := group * out_per_group + local_oc
					for k in 0 .. weight.shape[2] {
						output_pos := input_pos * config.stride - config.padding + k * config.dilation
						if output_pos >= 0 && output_pos < output_length {
							index := [b, oc, output_pos]
							output.set(index, output.get(index) + value * weight.get([
								ic,
								local_oc,
								k,
							]))
						}
					}
				}
			}
		}
	}
	for b in 0 .. batch {
		for oc in 0 .. out_channels {
			for output_pos in 0 .. output_length {
				index := [b, oc, output_pos]
				output.set(index, output.get(index) + bias.get_nth(oc))
			}
		}
	}
	return output
}

pub fn conv_transpose1d_backward[T](grad_output &vtl.Tensor[T], input &vtl.Tensor[T], weight &vtl.Tensor[T], bias &vtl.Tensor[T], config ConvTranspose1DConfig) ![]&vtl.Tensor[T] {
	output_length := validate_conv_transpose1d(input, weight, bias, config)!
	expected := [input.shape[0], weight.shape[1] * config.groups, output_length]
	if grad_output.shape != expected {
		return error('conv_transpose1d_backward: grad_output shape must be ${expected}, got ${grad_output.shape}')
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
			for input_pos in 0 .. input.shape[2] {
				for local_oc in 0 .. out_per_group {
					oc := group * out_per_group + local_oc
					for k in 0 .. weight.shape[2] {
						output_pos := input_pos * config.stride - config.padding + k * config.dilation
						if output_pos >= 0 && output_pos < output_length {
							grad := grad_output.get([b, oc, output_pos])
							d_input.set([b, ic, input_pos], d_input.get([b, ic, input_pos]) + grad * weight.get([
								ic,
								local_oc,
								k,
							]))
							d_weight.set([ic, local_oc, k], d_weight.get([ic, local_oc, k]) + grad * input.get([
								b,
								ic,
								input_pos,
							]))
						}
					}
				}
			}
		}
	}
	for b in 0 .. batch {
		for oc in 0 .. out_channels {
			for output_pos in 0 .. output_length {
				d_bias.set_nth(oc, d_bias.get_nth(oc) + grad_output.get([b, oc, output_pos]))
			}
		}
	}
	return [d_input, d_weight, d_bias]
}

fn validate_conv_transpose1d[T](input &vtl.Tensor[T], weight &vtl.Tensor[T], bias &vtl.Tensor[T], config ConvTranspose1DConfig) !int {
	if input.shape.len != 3 || weight.shape.len != 3 {
		return error('conv_transpose1d: input and weight must have rank 3')
	}
	if config.stride <= 0 || config.dilation <= 0 || config.groups <= 0 || config.padding < 0
		|| config.output_padding < 0 || (config.output_padding >= config.stride && config.output_padding >= config.dilation) {
		return error('conv_transpose1d: invalid stride, padding, output_padding, dilation, or groups')
	}
	in_channels := input.shape[1]
	out_channels := weight.shape[1] * config.groups
	if in_channels <= 0 || in_channels % config.groups != 0 || weight.shape[0] != in_channels
		|| weight.shape[1] <= 0 || bias.size() != out_channels {
		return error('conv_transpose1d: incompatible input, weight, bias, or group dimensions')
	}
	if input.shape[2] <= 0 || weight.shape[2] <= 0 {
		return error('conv_transpose1d: input length and kernel size must be positive')
	}
	output_length := (input.shape[2] - 1) * config.stride - 2 * config.padding +
		config.dilation * (weight.shape[2] - 1) + config.output_padding + 1
	if output_length <= 0 {
		return error('conv_transpose1d: calculated output length must be positive')
	}
	return output_length
}
