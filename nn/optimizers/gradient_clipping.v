module optimizers

import math
import vtl
import vtl.autograd

fn is_float_gradient[T]() bool {
	$if T is f32 || T is f64 {
		return true
	} $else {
		return false
	}
}

// clip_grad_norm scales trainable gradients in parameters in place when their
// global L2 norm exceeds max_norm. It returns the norm before clipping.
pub fn clip_grad_norm[T](mut parameters []&autograd.Variable[T], max_norm f64) !f64 {
	return clip_grad_norm_with_type[T](mut parameters, max_norm, 2.0)
}

// clip_grad_norm_with_type clips by a global p-norm; positive infinity selects
// the infinity norm. It returns the pre-clipping norm.
pub fn clip_grad_norm_with_type[T](mut parameters []&autograd.Variable[T], max_norm f64, norm_type f64) !f64 {
	if !is_float_gradient[T]() {
		return error('clip_grad_norm: gradients must use f32 or f64 elements')
	}
	if max_norm < 0.0 || !math.is_finite(max_norm) {
		return error('clip_grad_norm: max_norm must be finite and non-negative')
	}
	norm := get_total_grad_norm_with_type[T](parameters, norm_type)!
	clip_grads_with_norm[T](mut parameters, max_norm, norm)!
	return norm
}

// get_total_grad_norm computes the L2 norm of all trainable gradients without
// changing them.
pub fn get_total_grad_norm[T](parameters []&autograd.Variable[T]) !f64 {
	return get_total_grad_norm_with_type[T](parameters, 2.0)
}

// get_total_grad_norm_with_type computes a global p-norm without changing
// gradients; positive infinity selects the infinity norm.
pub fn get_total_grad_norm_with_type[T](parameters []&autograd.Variable[T], norm_type f64) !f64 {
	if !is_float_gradient[T]() {
		return error('get_total_grad_norm: gradients must use f32 or f64 elements')
	}
	if norm_type < 0.0 || (norm_type != math.inf(1) && !math.is_finite(norm_type)) {
		return error('get_total_grad_norm: norm_type must be finite and non-negative, or positive infinity')
	}

	mut norm_scale := 0.0
	mut scaled_squares := 1.0
	mut scaled_powers := 0.0
	for parameter in parameters {
		if !parameter.requires_grad {
			continue
		}
		for i in 0 .. parameter.grad.size {
			gradient := math.abs(f64(parameter.grad.get_nth(i)))
			if !math.is_finite(gradient) {
				return error('get_total_grad_norm: gradient norm must be finite')
			}
			if gradient == 0.0 {
				continue
			}
			if norm_type == 0.0 {
				scaled_powers += 1.0
			} else if norm_type == math.inf(1) {
				if norm_scale < gradient {
					norm_scale = gradient
				}
			} else if norm_type == 2.0 {
				if norm_scale < gradient {
					ratio := norm_scale / gradient
					scaled_squares = 1.0 + scaled_squares * ratio * ratio
					norm_scale = gradient
				} else {
					ratio := gradient / norm_scale
					scaled_squares += ratio * ratio
				}
			} else if norm_scale < gradient {
				ratio := norm_scale / gradient
				scaled_powers = 1.0 + scaled_powers * math.pow(ratio, norm_type)
				norm_scale = gradient
			} else {
				ratio := gradient / norm_scale
				scaled_powers += math.pow(ratio, norm_type)
			}
		}
	}
	mut norm := norm_scale
	if norm_type == 0.0 {
		norm = scaled_powers
	} else if norm_type == 2.0 {
		norm *= math.sqrt(scaled_squares)
	} else if norm_type != math.inf(1) && norm_scale > 0.0 {
		norm *= math.pow(scaled_powers, 1.0 / norm_type)
	}
	if !math.is_finite(norm) {
		return error('get_total_grad_norm: gradient norm must be finite')
	}
	return norm
}

// clip_grads_with_norm scales trainable gradients in place using a previously
// computed total norm.
pub fn clip_grads_with_norm[T](mut parameters []&autograd.Variable[T], max_norm f64, total_norm f64) ! {
	if !is_float_gradient[T]() {
		return error('clip_grads_with_norm: gradients must use f32 or f64 elements')
	}
	if max_norm < 0.0 || !math.is_finite(max_norm) {
		return error('clip_grads_with_norm: max_norm must be finite and non-negative')
	}
	if total_norm < 0.0 || !math.is_finite(total_norm) {
		return error('clip_grads_with_norm: total_norm must be finite and non-negative')
	}
	if total_norm <= max_norm || total_norm == 0.0 {
		return
	}
	scale := max_norm / total_norm
	for mut parameter in parameters {
		if !parameter.requires_grad {
			continue
		}
		parameter.grad.apply(fn [scale] [T](value T, _ []int) T {
			return vtl.cast[T](f64(value) * scale)
		})
	}
}

// clip_grad_value clamps every trainable gradient element to [-max_value, max_value].
pub fn clip_grad_value[T](mut parameters []&autograd.Variable[T], max_value f64) ! {
	if max_value < 0.0 || !math.is_finite(max_value) {
		return error('clip_grad_value: max_value must be finite and non-negative')
	}
	for mut parameter in parameters {
		if !parameter.requires_grad {
			continue
		}
		parameter.grad.apply(fn [max_value] [T](value T, _ []int) T {
			x := f64(value)
			if x < -max_value {
				return vtl.cast[T](-max_value)
			}
			if x > max_value {
				return vtl.cast[T](max_value)
			}
			return value
		})
	}
}
