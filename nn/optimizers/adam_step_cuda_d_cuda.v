module optimizers

import math
import vsl.cuda
import vsl.cuda.compute
import vtl.autograd

// adam_step_f64 runs Adam moment updates on GPU when cuda_optimizer_enabled; final
// bias-corrected step uses CPU sqrt (no GPU sqrt kernel in VSL yet).
pub fn adam_step_f64(grad []f64, mut theta []f64, mut m []f64, mut v []f64, p AdamStepParams) {
	if !autograd.cuda_optimizer_enabled() {
		adam_step_f64_cpu(grad, mut theta, mut m, mut v, p)
		return
	}
	dev := cuda.get_default_device() or {
		adam_step_f64_cpu(grad, mut theta, mut m, mut v, p)
		return
	}
	// m = beta1*m + (1-beta1)*g
	mut m_gpu := compute.mul_scalar_cuda(dev, m, p.beta1)!
	m_part := compute.mul_scalar_cuda(dev, grad, 1.0 - p.beta1)!
	m_gpu = compute.add_vec_cuda(dev, m_part, m_gpu)!
	// v = beta2*v + (1-beta2)*g^2
	g_sq := compute.mul_vec_cuda(dev, grad, grad)!
	mut v_gpu := compute.mul_scalar_cuda(dev, v, p.beta2)!
	v_part := compute.mul_scalar_cuda(dev, g_sq, 1.0 - p.beta2)!
	v_gpu = compute.add_vec_cuda(dev, v_part, v_gpu)!
	m = m_gpu
	v = v_gpu
	for i in 0 .. theta.len {
		theta[i] -= p.lr_t * m[i] / (math.sqrt(v[i]) + p.epsilon)
	}
}
