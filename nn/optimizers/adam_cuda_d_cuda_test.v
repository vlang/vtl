module optimizers

import math
import os
import vtl.autograd
import vtl.autograd_cuda

fn test_adam_step_cuda_matches_cpu_when_enabled() {
	if os.getenv('VTL_TEST_CUDA') != '1' || os.getenv('VTL_CUDA_OPTIMIZER') != '1' {
		return
	}
	grad := [1.0, 2.0, 0.5]
	mut theta_cpu := [3.0, 4.0, 5.0]
	mut m_cpu := [0.1, 0.2, 0.3]
	mut v_cpu := [0.01, 0.02, 0.03]
	mut theta_gpu := theta_cpu.clone()
	mut m_gpu := m_cpu.clone()
	mut v_gpu := v_cpu.clone()
	p := AdamStepParams{
		beta1:   0.9
		beta2:   0.999
		lr_t:    0.01
		epsilon: 1e-8
	}
	mut c := autograd.ctx[f64]()
	autograd_cuda.attach_context_session(mut c)
	adam_step_f64_cpu(grad, mut theta_cpu, mut m_cpu, mut v_cpu, p)
	adam_step_f64(grad, mut theta_gpu, mut m_gpu, mut v_gpu, p, c.device_session, 0)
	for i in 0 .. grad.len {
		assert math.abs(theta_cpu[i] - theta_gpu[i]) < 1e-6
		assert math.abs(m_cpu[i] - m_gpu[i]) < 1e-6
		assert math.abs(v_cpu[i] - v_gpu[i]) < 1e-6
	}
}

fn test_adam_step_persistent_gpu_second_step_matches_cpu() {
	if os.getenv('VTL_TEST_CUDA') != '1' || os.getenv('VTL_CUDA_OPTIMIZER') != '1' {
		return
	}
	mut c := autograd.ctx[f64]()
	autograd_cuda.attach_context_session(mut c)
	grad := [0.5, -0.25]
	p := AdamStepParams{
		beta1:   0.9
		beta2:   0.999
		lr_t:    0.01
		epsilon: 1e-8
	}
	mut th_cpu := [2.0, 3.0]
	mut m_cpu := [0.0, 0.0]
	mut v_cpu := [0.0, 0.0]
	mut th_gpu := th_cpu.clone()
	mut m_gpu := m_cpu.clone()
	mut v_gpu := v_cpu.clone()
	for _ in 0 .. 2 {
		adam_step_f64_cpu(grad, mut th_cpu, mut m_cpu, mut v_cpu, p)
		adam_step_f64(grad, mut th_gpu, mut m_gpu, mut v_gpu, p, c.device_session, 0)
	}
	for i in 0 .. grad.len {
		assert math.abs(th_cpu[i] - th_gpu[i]) < 1e-5
	}
}
