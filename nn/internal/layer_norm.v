module internal

import math
import vtl

// layer_norm_forward computes layer normalization over the last `ndim` dims of input.
// input: [..., D]  normalized_shape: [D]
// gamma, beta: [D] (optional affine params)
pub fn layer_norm_forward[T](input &vtl.Tensor[T], gamma &vtl.Tensor[T], beta &vtl.Tensor[T], eps f64) !&vtl.Tensor[T] {
	shape := input.shape
	ndim := shape.len
	D := vtl.prod(shape)

	// Compute mean over the normalized axes
	mut mean_data := []f64{len: 1}
	mut sum := f64(0)
	for i in 0 .. D {
		sum += f64(input.get_nth(i))
	}
	mean_data[0] = sum / f64(D)
	mean := vtl.from_1d(mean_data.map(vtl.cast[T](it)), [1])!

	// Compute variance
	mut var_sum := f64(0)
	for i in 0 .. D {
		diff := f64(input.get_nth(i)) - mean_data[0]
		var_sum += diff * diff
	}
	var_ := var_sum / f64(D)

	// Normalize: (x - mean) / sqrt(var + eps)
	std := 1.0 / math.sqrt(var_ + eps)
	mut output := vtl.zeros_like[T](input)
	for i in 0 .. D {
		normalized := (f64(input.get_nth(i)) - mean_data[0]) * std
		if gamma != unsafe { nil } {
			normalized *= f64(gamma.get_nth(i))
		}
		if beta != unsafe { nil } {
			normalized += f64(beta.get_nth(i))
		}
		output.set_nth(i, vtl.cast[T](normalized))
	}
	return output
}

// layer_norm_backward computes gradient w.r.t. input, gamma, beta.
pub fn layer_norm_backward[T](gradient &vtl.Tensor[T], input &vtl.Tensor[T], gamma &vtl.Tensor[T], beta &vtl.Tensor[T], eps f64) ![]&vtl.Tensor[T] {
	D := input.size()

	// Recompute mean and std
	mut sum_mean := f64(0)
	for i in 0 .. D { sum_mean += f64(input.get_nth(i)) }
	mean := sum_mean / f64(D)
	mut sum_var := f64(0)
	for i in 0 .. D {
		diff := f64(input.get_nth(i)) - mean
		sum_var += diff * diff
	}
	var_ := sum_var / f64(D)
	std := 1.0 / math.sqrt(var_ + eps)

	mut dx_data := []f64{len: D}
	mut dgamma_data := []f64{len: D}
	mut dbeta_data := []f64{len: D}

	for i in 0 .. D {
		norm := (f64(input.get_nth(i)) - mean) * std
		grad := f64(gradient.get_nth(i))
		if gamma != unsafe { nil } {
			g_val := f64(gamma.get_nth(i))
			dx_data[i] = grad * g_val * std
			dgamma_data[i] = grad * norm
		} else {
			dx_data[i] = grad * std
		}
		if beta != unsafe { nil } {
			dbeta_data[i] = grad
		}
	}

	dx := vtl.from_array(dx_data.map(vtl.cast[T](it)), input.shape)!
	dgamma := if gamma != unsafe { nil } { vtl.from_array(dgamma_data.map(vtl.cast[T](it)), gamma.shape)! } else { unsafe { nil } }
	dbeta := if beta != unsafe { nil } { vtl.from_array(dbeta_data.map(vtl.cast[T](it)), beta.shape)! } else { unsafe { nil } }

	if dgamma != unsafe { nil } && dbeta != unsafe { nil } {
		return [dx, dgamma, dbeta]
	}
	return [dx]
}