module internal

import math
import vtl

// layer_norm_forward computes layer normalization over all elements of input.
// gamma, beta: same shape as input (optional affine params, pass nil to skip)
pub fn layer_norm_forward[T](input &vtl.Tensor[T], gamma &vtl.Tensor[T], beta &vtl.Tensor[T], eps f64) !&vtl.Tensor[T] {
	d_size := input.size()

	// Compute mean
	mut sum := f64(0)
	for i in 0 .. d_size {
		sum += f64(input.get_nth(i))
	}
	mean_val := sum / f64(d_size)

	// Compute variance
	mut var_sum := f64(0)
	for i in 0 .. d_size {
		diff := f64(input.get_nth(i)) - mean_val
		var_sum += diff * diff
	}
	var_ := var_sum / f64(d_size)

	// Normalize: (x - mean) / sqrt(var + eps)
	std := 1.0 / math.sqrt(var_ + eps)
	mut output := vtl.zeros_like[T](input)
	for i in 0 .. d_size {
		mut normalized := (f64(input.get_nth(i)) - mean_val) * std
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
	d_size := input.size()

	// Recompute mean and std
	mut sum_mean := f64(0)
	for i in 0 .. d_size {
		sum_mean += f64(input.get_nth(i))
	}
	mean := sum_mean / f64(d_size)
	mut sum_var := f64(0)
	for i in 0 .. d_size {
		diff := f64(input.get_nth(i)) - mean
		sum_var += diff * diff
	}
	var_ := sum_var / f64(d_size)
	std := 1.0 / math.sqrt(var_ + eps)

	mut dx_data := []f64{len: d_size}
	mut dgamma_data := []f64{len: d_size}
	mut dbeta_data := []f64{len: d_size}

	for i in 0 .. d_size {
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

	if gamma != unsafe { nil } && beta != unsafe { nil } {
		dgamma := vtl.from_array(dgamma_data.map(vtl.cast[T](it)), gamma.shape)!
		dbeta := vtl.from_array(dbeta_data.map(vtl.cast[T](it)), beta.shape)!
		return [dx, dgamma, dbeta]
	}
	return [dx]
}
