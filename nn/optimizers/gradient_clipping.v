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
	if norm <= max_norm || norm == 0.0 {
		return norm
	}

	scale := vtl.cast[T](max_norm / norm)
	for mut parameter in parameters {
		if parameter.requires_grad {
			parameter.grad = parameter.grad.multiply_scalar[T](scale)!
		}
	}
	return norm
}
