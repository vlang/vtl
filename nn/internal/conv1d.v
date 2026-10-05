module internal

import vtl

// Conv1DConfig configures a one-dimensional convolution.
pub struct Conv1DConfig {
pub:
	stride   int = 1
	padding  int
	dilation int = 1
	groups   int = 1
}

// conv1d_forward computes grouped cross-correlation over input [batch, channels, length].
// Weight layout is [out_channels, in_channels/groups, kernel_size].
pub fn conv1d_forward[T](input &vtl.Tensor[T], weight &vtl.Tensor[T], bias &vtl.Tensor[T],
	config Conv1DConfig) !&vtl.Tensor[T] {
	validate_conv1d(input, weight, bias, config)!
	batch, in_channels, length := input.shape[0], input.shape[1], input.shape[2]
	out_channels, kernel_size := weight.shape[0], weight.shape[2]
	out_length := conv1d_output_length(length, kernel_size, config)!
	in_per_group := in_channels / config.groups
	out_per_group := out_channels / config.groups
	mut output := vtl.zeros[T]([batch, out_channels, out_length])
	for b in 0 .. batch {
		for oc in 0 .. out_channels {
			group := oc / out_per_group
			for pos in 0 .. out_length {
				mut sum := vtl.cast[T](0)
				for local_ic in 0 .. in_per_group {
					ic := group * in_per_group + local_ic
					for k in 0 .. kernel_size {
						index := pos * config.stride - config.padding + k * config.dilation
						if index >= 0 && index < length {
							sum += input.get([b, ic, index]) * weight.get([oc, local_ic, k])
						}
					}
				}
				output.set([b, oc, pos], sum + bias.get_nth(oc))
			}
		}
	}
	return output
}

// conv1d_backward computes derivatives for input, weight, and bias.
pub fn conv1d_backward[T](grad_output &vtl.Tensor[T], input &vtl.Tensor[T], weight &vtl.Tensor[T],
	bias &vtl.Tensor[T], config Conv1DConfig) ![]&vtl.Tensor[T] {
	validate_conv1d(input, weight, bias, config)!
	out_length := conv1d_output_length(input.shape[2], weight.shape[2], config)!
	expected := [input.shape[0], weight.shape[0], out_length]
	if grad_output.shape != expected {
		return error('conv1d_backward: grad_output shape must be ${expected}, got ${grad_output.shape}')
	}
	in_channels, out_channels := input.shape[1], weight.shape[0]
	in_per_group := in_channels / config.groups
	out_per_group := out_channels / config.groups
	mut d_input := vtl.zeros_like[T](input)
	mut d_weight := vtl.zeros_like[T](weight)
	mut d_bias := vtl.zeros_like[T](bias)
	for b in 0 .. input.shape[0] {
		for oc in 0 .. out_channels {
			group := oc / out_per_group
			for pos in 0 .. out_length {
				grad := grad_output.get([b, oc, pos])
				d_bias.set_nth(oc, d_bias.get_nth(oc) + grad)
				for local_ic in 0 .. in_per_group {
					ic := group * in_per_group + local_ic
					for k in 0 .. weight.shape[2] {
						index := pos * config.stride - config.padding + k * config.dilation
						if index >= 0 && index < input.shape[2] {
							d_input.set([b, ic, index], d_input.get([b, ic, index]) + grad * weight.get([
								oc,
								local_ic,
								k,
							]))
							d_weight.set([oc, local_ic, k], d_weight.get([oc, local_ic, k]) + grad * input.get([
								b,
								ic,
								index,
							]))
						}
					}
				}
			}
		}
	}
	return [d_input, d_weight, d_bias]
}

fn validate_conv1d[T](input &vtl.Tensor[T], weight &vtl.Tensor[T], bias &vtl.Tensor[T],
	config Conv1DConfig) ! {
	if input.shape.len != 3 || weight.shape.len != 3 {
		return error('conv1d: input and weight must have rank 3')
	}
	if config.stride <= 0 || config.padding < 0 || config.dilation <= 0 || config.groups <= 0 {
		return error('conv1d: stride/dilation/groups must be positive and padding non-negative')
	}
	if input.shape[1] % config.groups != 0 || weight.shape[0] % config.groups != 0
		|| weight.shape[1] != input.shape[1] / config.groups || bias.size() != weight.shape[0] {
		return error('conv1d: incompatible input, weight, bias, or group dimensions')
	}
	_ := conv1d_output_length(input.shape[2], weight.shape[2], config)!
}

fn conv1d_output_length(length int, kernel_size int, config Conv1DConfig) !int {
	if length < 0 || kernel_size <= 0 {
		return error('conv1d: input length must be non-negative and kernel_size positive')
	}
	span := config.dilation * (kernel_size - 1) + 1
	if length + 2 * config.padding < span {
		return error('conv1d: effective kernel is larger than the padded input')
	}
	return (length + 2 * config.padding - span) / config.stride + 1
}
