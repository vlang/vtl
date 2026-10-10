module internal

import math
import vtl

// layer_norm_forward computes layer normalization over all input elements.
// gamma, beta: optional affine parameters matching the normalized trailing shape
pub fn layer_norm_forward[T](input &vtl.Tensor[T], gamma &vtl.Tensor[T], beta &vtl.Tensor[T], eps f64) !&vtl.Tensor[T] {
	normalized_shape := layer_norm_default_shape(input, gamma, beta)
	return layer_norm_forward_shape[T](input, gamma, beta, normalized_shape, eps)
}

// layer_norm_forward_shape normalizes independently over each trailing block
// described by normalized_shape, matching the LayerNorm definition used by
// PyTorch. Affine parameters have exactly normalized_shape and are broadcast
// over all leading dimensions.
pub fn layer_norm_forward_shape[T](input &vtl.Tensor[T], gamma &vtl.Tensor[T], beta &vtl.Tensor[T], normalized_shape []int, eps f64) !&vtl.Tensor[T] {
	normalized_size := layer_norm_validate(input, gamma, beta, normalized_shape, eps)!
	group_count := input.size / normalized_size
	mut output := vtl.zeros_like[T](input)
	for group in 0 .. group_count {
		start := group * normalized_size
		mut mean := 0.0
		for offset in 0 .. normalized_size {
			mean += f64(input.get_nth(start + offset))
		}
		mean /= f64(normalized_size)

		mut variance := 0.0
		for offset in 0 .. normalized_size {
			diff := f64(input.get_nth(start + offset)) - mean
			variance += diff * diff
		}
		variance /= f64(normalized_size)
		inv_std := 1.0 / math.sqrt(variance + eps)

		for offset in 0 .. normalized_size {
			index := start + offset
			mut value := (f64(input.get_nth(index)) - mean) * inv_std
			if gamma != unsafe { nil } {
				value *= f64(gamma.get_nth(offset))
			}
			if beta != unsafe { nil } {
				value += f64(beta.get_nth(offset))
			}
			output.set_nth(index, vtl.cast[T](value))
		}
	}
	return output
}

// layer_norm_backward computes gradients using the full input shape as the
// normalized shape, preserving the original low-level API behavior.
pub fn layer_norm_backward[T](gradient &vtl.Tensor[T], input &vtl.Tensor[T], gamma &vtl.Tensor[T], beta &vtl.Tensor[T], eps f64) ![]&vtl.Tensor[T] {
	normalized_shape := layer_norm_default_shape(input, gamma, beta)
	return layer_norm_backward_shape[T](gradient, input, gamma, beta, normalized_shape, eps)
}

// layer_norm_backward_shape computes input and affine gradients over trailing
// normalized_shape dimensions. Affine gradients reduce across leading groups.
pub fn layer_norm_backward_shape[T](gradient &vtl.Tensor[T], input &vtl.Tensor[T], gamma &vtl.Tensor[T], beta &vtl.Tensor[T], normalized_shape []int, eps f64) ![]&vtl.Tensor[T] {
	normalized_size := layer_norm_validate(input, gamma, beta, normalized_shape, eps)!
	if gradient.shape != input.shape {
		return error('layer_norm_backward: gradient shape must match input shape')
	}
	group_count := input.size / normalized_size
	mut dx_data := []f64{len: input.size}
	mut dgamma_data := if gamma != unsafe { nil } { []f64{len: normalized_size} } else { []f64{} }
	mut dbeta_data := if beta != unsafe { nil } { []f64{len: normalized_size} } else { []f64{} }

	for group in 0 .. group_count {
		start := group * normalized_size
		mut mean := 0.0
		for offset in 0 .. normalized_size {
			mean += f64(input.get_nth(start + offset))
		}
		mean /= f64(normalized_size)

		mut variance := 0.0
		for offset in 0 .. normalized_size {
			diff := f64(input.get_nth(start + offset)) - mean
			variance += diff * diff
		}
		variance /= f64(normalized_size)
		inv_std := 1.0 / math.sqrt(variance + eps)

		mut mean_scaled_gradient := 0.0
		mut mean_scaled_gradient_xhat := 0.0
		for offset in 0 .. normalized_size {
			index := start + offset
			xhat := (f64(input.get_nth(index)) - mean) * inv_std
			grad := f64(gradient.get_nth(index))
			gamma_value := if gamma != unsafe { nil } { f64(gamma.get_nth(offset)) } else { 1.0 }
			scaled_grad := grad * gamma_value
			mean_scaled_gradient += scaled_grad
			mean_scaled_gradient_xhat += scaled_grad * xhat
			if gamma != unsafe { nil } {
				dgamma_data[offset] += grad * xhat
			}
			if beta != unsafe { nil } {
				dbeta_data[offset] += grad
			}
		}
		mean_scaled_gradient /= f64(normalized_size)
		mean_scaled_gradient_xhat /= f64(normalized_size)

		for offset in 0 .. normalized_size {
			index := start + offset
			xhat := (f64(input.get_nth(index)) - mean) * inv_std
			scaled_grad := f64(gradient.get_nth(index)) * if gamma != unsafe { nil } {
				f64(gamma.get_nth(offset))
			} else {
				1.0
			}
			dx_data[index] = inv_std * (scaled_grad - mean_scaled_gradient - xhat * mean_scaled_gradient_xhat)
		}
	}

	dx := vtl.from_array(dx_data.map(vtl.cast[T](it)), input.shape)!
	if gamma != unsafe { nil } {
		dgamma := vtl.from_array(dgamma_data.map(vtl.cast[T](it)), normalized_shape)!
		if beta == unsafe { nil } {
			return [dx, dgamma]
		}
		dbeta := vtl.from_array(dbeta_data.map(vtl.cast[T](it)), normalized_shape)!
		return [dx, dgamma, dbeta]
	}
	if beta != unsafe { nil } {
		dbeta := vtl.from_array(dbeta_data.map(vtl.cast[T](it)), normalized_shape)!
		return [dx, dbeta]
	}
	return [dx]
}

fn layer_norm_default_shape[T](input &vtl.Tensor[T], gamma &vtl.Tensor[T], beta &vtl.Tensor[T]) []int {
	if gamma != unsafe { nil } {
		return gamma.shape.clone()
	}
	if beta != unsafe { nil } {
		return beta.shape.clone()
	}
	return input.shape.clone()
}

fn layer_norm_validate[T](input &vtl.Tensor[T], gamma &vtl.Tensor[T], beta &vtl.Tensor[T], normalized_shape []int, eps f64) !int {
	if normalized_shape.len == 0 || normalized_shape.len > input.rank() {
		return error('layer_norm: normalized_shape must contain between 1 and input rank dimensions')
	}
	if !math.is_finite(eps) || eps < 0.0 {
		return error('layer_norm: eps must be finite and non-negative')
	}
	mut normalized_size := 1
	start_axis := input.rank() - normalized_shape.len
	for i, dimension in normalized_shape {
		if dimension <= 0 || input.shape[start_axis + i] != dimension {
			return error('layer_norm: normalized_shape must match the trailing input dimensions')
		}
		if normalized_size > max_int / dimension {
			return error('layer_norm: normalized_shape size overflows int')
		}
		normalized_size *= dimension
	}
	if normalized_size <= 0 || input.size % normalized_size != 0 {
		return error('layer_norm: input size must be divisible by normalized shape size')
	}
	if gamma != unsafe { nil } && gamma.shape != normalized_shape {
		return error('layer_norm: gamma shape must match normalized_shape')
	}
	if beta != unsafe { nil } && beta.shape != normalized_shape {
		return error('layer_norm: beta shape must match normalized_shape')
	}
	return normalized_size
}
