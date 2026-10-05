module optimizers

import math

// radam_step_f32_cpu applies one Rectified Adam update in f32.
pub fn radam_step_f32_cpu(grad []f32, mut theta []f32, mut m []f32, mut v []f32, p RAdamStepParams) {
	b1 := f32(p.beta1)
	b2 := f32(p.beta2)
	b1_t := f32(p.beta1_t)
	b2_t := f32(p.beta2_t)
	lr := f32(p.lr)
	eps := f32(p.epsilon)
	wd := f32(p.weight_decay)
	rho_inf := 2.0 / (1.0 - p.beta2) - 1.0
	rho_t := rho_inf - 2.0 * f64(p.step) * p.beta2_t / (1.0 - p.beta2_t)
	for i in 0 .. grad.len {
		if p.decoupled_decay { theta[i] -= lr * wd * theta[i] }
		g := if p.decoupled_decay { grad[i] } else { grad[i] + wd * theta[i] }
		m[i] = b1 * m[i] + (1.0 - b1) * g
		v[i] = b2 * v[i] + (1.0 - b2) * g * g
		m_hat := m[i] / (1.0 - b1_t)
		if rho_t > 5.0 {
			rectification := math.sqrt(((rho_t - 4.0) * (rho_t - 2.0) * rho_inf) / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t))
			step_size := lr * f32(rectification) * f32(math.sqrt(1.0 - f64(b2_t)))
			theta[i] -= step_size * m_hat / (f32(math.sqrt(f64(v[i]))) + eps)
		} else {
			theta[i] -= lr * m_hat
		}
	}
}
