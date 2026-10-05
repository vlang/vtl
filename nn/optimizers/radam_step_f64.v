module optimizers

import math

// RAdamStepParams holds one CPU RAdam update's scalar state.
pub struct RAdamStepParams {
pub:
	beta1           f64
	beta2           f64
	beta1_t         f64
	beta2_t         f64
	step            int
	lr              f64
	epsilon         f64
	weight_decay    f64
	decoupled_decay bool
}

// radam_step_f64_cpu applies one Rectified Adam update in f64.
pub fn radam_step_f64_cpu(grad []f64, mut theta []f64, mut m []f64, mut v []f64, p RAdamStepParams) {
	rho_inf := 2.0 / (1.0 - p.beta2) - 1.0
	rho_t := rho_inf - 2.0 * f64(p.step) * p.beta2_t / (1.0 - p.beta2_t)
	for i in 0 .. grad.len {
		if p.decoupled_decay { theta[i] -= p.lr * p.weight_decay * theta[i] }
		mut g := grad[i]
		if !p.decoupled_decay { g += p.weight_decay * theta[i] }
		m[i] = p.beta1 * m[i] + (1.0 - p.beta1) * g
		v[i] = p.beta2 * v[i] + (1.0 - p.beta2) * g * g
		m_hat := m[i] / (1.0 - p.beta1_t)
		if rho_t > 5.0 {
			rectification := math.sqrt(((rho_t - 4.0) * (rho_t - 2.0) * rho_inf) / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t))
			step_size := p.lr * rectification * math.sqrt(1.0 - p.beta2_t)
			theta[i] -= step_size * m_hat / (math.sqrt(v[i]) + p.epsilon)
		} else {
			theta[i] -= p.lr * m_hat
		}
	}
}
