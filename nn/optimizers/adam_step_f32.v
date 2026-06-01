module optimizers

import math

// adam_step_f32_cpu performs one Adam update on flat f32 buffers (same math as napply path).
pub fn adam_step_f32_cpu(grad []f32, mut theta []f32, mut m []f32, mut v []f32, p AdamStepParams) {
	n := grad.len
	b1 := f32(p.beta1)
	b2 := f32(p.beta2)
	lr := f32(p.lr_t)
	eps := f32(p.epsilon)
	one_minus_b1 := f32(1.0 - p.beta1)
	one_minus_b2 := f32(1.0 - p.beta2)
	for i in 0 .. n {
		g := grad[i]
		m[i] = b1 * m[i] + one_minus_b1 * g
		v[i] = b2 * v[i] + one_minus_b2 * g * g
		denom := f32(math.sqrt(f64(v[i]))) + eps
		theta[i] -= lr * m[i] / denom
	}
}
