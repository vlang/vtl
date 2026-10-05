module optimizers

import math

// NAdamStepParams holds one CPU NAdam update's scalar state.
pub struct NAdamStepParams {
pub:
	beta1           f64
	beta2           f64
	beta1_t         f64
	beta2_t         f64
	mu_product_t    f64
	mu_product_t1   f64
	mu_t            f64
	mu_next         f64
	momentum_decay  f64
	step            int
	lr              f64
	epsilon         f64
	weight_decay    f64
	decoupled_decay bool
}

// nadam_step_f64_cpu applies one Dozat NAdam update in f64.
pub fn nadam_step_f64_cpu(grad []f64, mut theta []f64, mut m []f64, mut v []f64, p NAdamStepParams) {
	for i in 0 .. grad.len {
		if p.decoupled_decay { theta[i] -= p.lr * p.weight_decay * theta[i] }
		mut g := grad[i]
		if !p.decoupled_decay { g += p.weight_decay * theta[i] }
		m[i] = p.beta1 * m[i] + (1.0 - p.beta1) * g
		v[i] = p.beta2 * v[i] + (1.0 - p.beta2) * g * g
		m_hat := f64(p.mu_next) * m[i] / (1.0 - p.mu_product_t1) + (1.0 - p.mu_t) * g / (1.0 - p.mu_product_t)
		v_hat := v[i] / (1.0 - p.beta2_t)
		theta[i] -= p.lr * m_hat / (math.sqrt(v_hat) + p.epsilon)
	}
}
