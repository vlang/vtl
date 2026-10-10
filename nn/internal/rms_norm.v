module internal

import math
import vtl

// rms_norm_forward normalizes each trailing normalized_shape block by its root
// mean square and optionally applies a per-element weight.
pub fn rms_norm_forward[T](input &vtl.Tensor[T], weight &vtl.Tensor[T], normalized_shape []int, eps f64) !&vtl.Tensor[T] {
	normalized_size := rms_norm_validate(input, weight, normalized_shape, eps)!
	input_data := input.to_array()
	weight_data := if weight != unsafe { nil } { weight.to_array() } else { []T{} }
	mut output := []T{len: input_data.len}
	for block in 0 .. input_data.len / normalized_size {
		mut square_sum := 0.0
		base := block * normalized_size
		for i in 0 .. normalized_size {
			value := f64(input_data[base + i])
			square_sum += value * value
		}
		inv_rms := 1.0 / math.sqrt(square_sum / f64(normalized_size) + eps)
		for i in 0 .. normalized_size {
			weight_value := if weight != unsafe { nil } { f64(weight_data[i]) } else { 1.0 }
			output[base + i] = vtl.cast[T](f64(input_data[base + i]) * inv_rms * weight_value)
		}
	}
	return vtl.from_array(output, input.shape.clone())
}

// rms_norm_backward computes input and optional per-element weight gradients.
pub fn rms_norm_backward[T](gradient &vtl.Tensor[T], input &vtl.Tensor[T], weight &vtl.Tensor[T], normalized_shape []int, eps f64) ![]&vtl.Tensor[T] {
	normalized_size := rms_norm_validate(input, weight, normalized_shape, eps)!
	if gradient.shape != input.shape {
		return error('rms_norm_backward: gradient shape must match input shape')
	}
	input_data := input.to_array()
	gradient_data := gradient.to_array()
	weight_data := if weight != unsafe { nil } { weight.to_array() } else { []T{} }
	mut dx := []f64{len: input_data.len}
	mut dweight := []f64{len: normalized_size}
	for block in 0 .. input_data.len / normalized_size {
		base := block * normalized_size
		mut square_sum := 0.0
		for i in 0 .. normalized_size {
			value := f64(input_data[base + i])
			square_sum += value * value
		}
		inv_rms := 1.0 / math.sqrt(square_sum / f64(normalized_size) + eps)
		mut weighted_dot := 0.0
		for i in 0 .. normalized_size {
			value := f64(input_data[base + i])
			upstream := f64(gradient_data[base + i])
			weight_value := if weight != unsafe { nil } { f64(weight_data[i]) } else { 1.0 }
			weighted_dot += upstream * weight_value * value
			dweight[i] += upstream * value * inv_rms
		}
		correction := inv_rms * inv_rms * inv_rms * weighted_dot / f64(normalized_size)
		for i in 0 .. normalized_size {
			upstream := f64(gradient_data[base + i])
			weight_value := if weight != unsafe { nil } { f64(weight_data[i]) } else { 1.0 }
			dx[base + i] = inv_rms * upstream * weight_value - f64(input_data[base + i]) * correction
		}
	}
	mut gradients := []&vtl.Tensor[T]{}
	gradients << vtl.from_array(dx.map(vtl.cast[T](it)), input.shape.clone())!
	if weight != unsafe { nil } {
		gradients << vtl.from_array(dweight.map(vtl.cast[T](it)), normalized_shape.clone())!
	}
	return gradients
}

fn rms_norm_validate[T](input &vtl.Tensor[T], weight &vtl.Tensor[T], normalized_shape []int, eps f64) !int {
	if normalized_shape.len == 0 || normalized_shape.len > input.shape.len {
		return error('rms_norm: normalized_shape must contain between one and input rank dimensions')
	}
	if !math.is_finite(eps) || eps < 0 {
		return error('rms_norm: eps must be finite and non-negative')
	}
	mut normalized_size := 1
	for i, dim in normalized_shape {
		if dim <= 0 || input.shape[input.shape.len - normalized_shape.len + i] != dim {
			return error('rms_norm: normalized_shape must match the trailing input dimensions')
		}
		if normalized_size > max_int / dim {
			return error('rms_norm: normalized_shape size overflows int')
		}
		normalized_size *= dim
	}
	if weight != unsafe { nil } && weight.shape != normalized_shape {
		return error('rms_norm: weight shape must match normalized_shape')
	}
	if input.size % normalized_size != 0 {
		return error('rms_norm: input size must be divisible by normalized_shape size')
	}
	return normalized_size
}
