module internal

import math
import vtl

// group_norm_forward normalizes each sample over channel groups and all
// trailing spatial dimensions. Affine parameters, when present, are per-channel.
pub fn group_norm_forward[T](input &vtl.Tensor[T], num_groups int, gamma &vtl.Tensor[T], beta &vtl.Tensor[T], eps f64) !&vtl.Tensor[T] {
	batch, channels, spatial, group_size := group_norm_validate(input, num_groups, gamma, beta,
		eps)!
	input_data := input.to_array()
	gamma_data := if gamma != unsafe { nil } { gamma.to_array() } else { []T{} }
	beta_data := if beta != unsafe { nil } { beta.to_array() } else { []T{} }
	mut output := []f64{len: input_data.len}
	channels_per_group := channels / num_groups
	for n in 0 .. batch {
		for group in 0 .. num_groups {
			channel_start := group * channels_per_group
			mut mean := f64(0)
			for channel in channel_start .. channel_start + channels_per_group {
				for pos in 0 .. spatial {
					index := (n * channels + channel) * spatial + pos
					mean += f64(input_data[index])
				}
			}
			mean /= f64(group_size)
			mut variance := f64(0)
			for channel in channel_start .. channel_start + channels_per_group {
				for pos in 0 .. spatial {
					index := (n * channels + channel) * spatial + pos
					difference := f64(input_data[index]) - mean
					variance += difference * difference
				}
			}
			inv_std := 1.0 / math.sqrt(variance / f64(group_size) + eps)
			for channel in channel_start .. channel_start + channels_per_group {
				weight := if gamma != unsafe { nil } { f64(gamma_data[channel]) } else { 1.0 }
				bias := if beta != unsafe { nil } { f64(beta_data[channel]) } else { 0.0 }
				for pos in 0 .. spatial {
					index := (n * channels + channel) * spatial + pos
					normalized := (f64(input_data[index]) - mean) * inv_std
					output[index] = weight * normalized + bias
				}
			}
		}
	}
	return vtl.from_array(output.map(vtl.cast[T](it)), input.shape.clone())
}

// group_norm_backward computes gradients for input and any present affine parameters.
pub fn group_norm_backward[T](gradient &vtl.Tensor[T], input &vtl.Tensor[T], num_groups int, gamma &vtl.Tensor[T], beta &vtl.Tensor[T], eps f64) ![]&vtl.Tensor[T] {
	batch, channels, spatial, group_size := group_norm_validate(input, num_groups, gamma, beta,
		eps)!
	if gradient.shape != input.shape {
		return error('group_norm_backward: gradient shape must match input shape')
	}
	input_data := input.to_array()
	gradient_data := gradient.to_array()
	gamma_data := if gamma != unsafe { nil } { gamma.to_array() } else { []T{} }
	mut dx := []f64{len: input_data.len}
	mut dgamma := []f64{len: channels}
	mut dbeta := []f64{len: channels}
	channels_per_group := channels / num_groups
	for n in 0 .. batch {
		for group in 0 .. num_groups {
			channel_start := group * channels_per_group
			mut mean := f64(0)
			for channel in channel_start .. channel_start + channels_per_group {
				for pos in 0 .. spatial {
					index := (n * channels + channel) * spatial + pos
					mean += f64(input_data[index])
				}
			}
			mean /= f64(group_size)
			mut variance := f64(0)
			for channel in channel_start .. channel_start + channels_per_group {
				for pos in 0 .. spatial {
					index := (n * channels + channel) * spatial + pos
					difference := f64(input_data[index]) - mean
					variance += difference * difference
				}
			}
			inv_std := 1.0 / math.sqrt(variance / f64(group_size) + eps)
			mut sum_dy := f64(0)
			mut sum_dy_xhat := f64(0)
			for channel in channel_start .. channel_start + channels_per_group {
				weight := if gamma != unsafe { nil } { f64(gamma_data[channel]) } else { 1.0 }
				for pos in 0 .. spatial {
					index := (n * channels + channel) * spatial + pos
					normalized := (f64(input_data[index]) - mean) * inv_std
					dy := f64(gradient_data[index])
					dygamma := dy * weight
					sum_dy += dygamma
					sum_dy_xhat += dygamma * normalized
					dgamma[channel] += dy * normalized
					dbeta[channel] += dy
				}
			}
			for channel in channel_start .. channel_start + channels_per_group {
				weight := if gamma != unsafe { nil } { f64(gamma_data[channel]) } else { 1.0 }
				for pos in 0 .. spatial {
					index := (n * channels + channel) * spatial + pos
					normalized := (f64(input_data[index]) - mean) * inv_std
					dygamma := f64(gradient_data[index]) * weight
					dx[index] = inv_std * (dygamma - sum_dy / f64(group_size) - normalized * sum_dy_xhat / f64(group_size))
				}
			}
		}
	}
	mut gradients := []&vtl.Tensor[T]{}
	gradients << vtl.from_array(dx.map(vtl.cast[T](it)), input.shape.clone())!
	if gamma != unsafe { nil } {
		gradients << vtl.from_array(dgamma.map(vtl.cast[T](it)), [channels])!
	}
	if beta != unsafe { nil } {
		gradients << vtl.from_array(dbeta.map(vtl.cast[T](it)), [channels])!
	}
	return gradients
}

fn group_norm_validate[T](input &vtl.Tensor[T], num_groups int, gamma &vtl.Tensor[T], beta &vtl.Tensor[T], eps f64) !(int, int, int, int) {
	if input.shape.len < 2 {
		return error('group_norm: input must have at least batch and channel dimensions')
	}
	if num_groups <= 0 {
		return error('group_norm: num_groups must be positive')
	}
	batch := input.shape[0]
	channels := input.shape[1]
	if channels <= 0 || channels % num_groups != 0 {
		return error('group_norm: num_groups must divide a positive channel count')
	}
	if !math.is_finite(eps) || eps < 0 {
		return error('group_norm: eps must be finite and non-negative')
	}
	if gamma != unsafe { nil } && gamma.shape != [channels] {
		return error('group_norm: gamma shape must match the channel count')
	}
	if beta != unsafe { nil } && beta.shape != [channels] {
		return error('group_norm: beta shape must match the channel count')
	}
	mut spatial := 1
	for dimension in input.shape[2..] {
		if dimension <= 0 || spatial > max_int / dimension {
			return error('group_norm: spatial dimensions must be positive and fit in int')
		}
		spatial *= dimension
	}
	channels_per_group := channels / num_groups
	if spatial > max_int / channels_per_group {
		return error('group_norm: group size overflows int')
	}
	group_size := channels_per_group * spatial
	if batch <= 0 {
		return error('group_norm: batch dimension must be positive')
	}
	return batch, channels, spatial, group_size
}
