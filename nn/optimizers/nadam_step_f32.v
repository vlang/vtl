module optimizers

import math

// nadam_step_f32_cpu applies one Dozat NAdam update in f32.
pub fn nadam_step_f32_cpu(grad []f32, mut theta []f32, mut m []f32, mut v []f32, p NAdamStepParams) {
	b1 := f32(p.beta1)
	b2 := f32(p.beta2)
	b2_t := f32(p.beta2_t)
	lr := f32(p.lr)
	eps := f32(p.epsilon)
	wd := f32(p.weight_decay)
	for i in 0 .. grad.len {
		if p.decoupled_decay { theta[i] -= lr * wd * theta[i] }
		g := if p.decoupled_decay { grad[i] } else { grad[i] + wd * theta[i] }
		m[i] = b1 * m[i] + (1.0 - b1) * g
		v[i] = b2 * v[i] + (1.0 - b2) * g * g
		m_hat := f32(p.mu_next) * m[i] / (1.0 - f32(p.mu_product_t1)) + (1.0 - f32(p.mu_t)) * g / (1.0 - f32(p.mu_product_t))
		v_hat := v[i] / (1.0 - b2_t)
		theta[i] -= lr * m_hat / (f32(math.sqrt(f64(v_hat))) + eps)
	}
}
