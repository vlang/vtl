module optimizers

import math

// AdamStepParams holds scalar Adam state for one flat parameter vector.
pub struct AdamStepParams {
pub:
	beta1     f64
	beta2     f64
	lr_t      f64
	epsilon   f64
}

// adam_step_f64_cpu performs one Adam update on flat CPU buffers.
pub fn adam_step_f64_cpu(grad []f64, mut theta []f64, mut m []f64, mut v []f64, p AdamStepParams) {
	n := grad.len
	for i in 0 .. n {
		g := grad[i]
		m[i] = p.beta1 * m[i] + (1.0 - p.beta1) * g
		v[i] = p.beta2 * v[i] + (1.0 - p.beta2) * g * g
		theta[i] -= p.lr_t * m[i] / (math.sqrt(v[i]) + p.epsilon)
	}
}
