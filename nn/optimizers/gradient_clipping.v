module optimizers

import math
import vtl
import vtl.autograd

// clip_grad_norm scales trainable gradients in parameters in place when their
// global L2 norm exceeds max_norm. It returns the norm before clipping.
pub fn clip_grad_norm[T](mut parameters []&autograd.Variable[T], max_norm f64) !f64 {
	if max_norm <= 0.0 || !math.is_finite(max_norm) {
		return error('clip_grad_norm: max_norm must be finite and greater than zero')
	}

	mut squared_norm := 0.0
	for parameter in parameters {
		if !parameter.requires_grad {
			continue
		}
		for i in 0 .. parameter.grad.size {
			gradient := f64(parameter.grad.get_nth(i))
			squared_norm += gradient * gradient
		}
	}
	norm := math.sqrt(squared_norm)
	if !math.is_finite(norm) {
		return error('clip_grad_norm: gradient norm must be finite')
	}
	if norm <= max_norm || norm == 0.0 {
		return norm
	}

	scale := max_norm / norm
	for mut parameter in parameters {
		if !parameter.requires_grad {
			continue
		}
		parameter.grad.apply(fn [scale] [T](value T, _ []int) T {
			return vtl.cast[T](f64(value) * scale)
		})
	}
	return norm
}

// clip_grad_value clamps every trainable gradient element to [-max_value, max_value].
pub fn clip_grad_value[T](mut parameters []&autograd.Variable[T], max_value f64) ! {
	if max_value <= 0.0 || !math.is_finite(max_value) {
		return error('clip_grad_value: max_value must be finite and greater than zero')
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
