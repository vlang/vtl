module optimizers

import math
import os
import vtl
import vtl.autograd
import vtl.autograd_cuda

fn ctx[T]() &autograd.Context[T] {
	return autograd.ctx[T]()
}

fn make_var[T](c &autograd.Context[T], val f64) &autograd.Variable[T] {
	t := vtl.from_1d([vtl.cast[T](val)]) or { panic(err) }
	return c.variable(t)
}

// SGD: param -= lr * grad
fn test_sgd_update() ! {
	c := ctx[f64]()
	mut p := make_var[f64](c, 5.0)
	mut opt := sgd[f64](learning_rate: 0.1)
	opt.params << p
	p.grad.set_nth(0, 1.0)
	opt.update()!
	new_val := p.value.get_nth(0)
	// 5.0 - 0.1 * 1.0 = 4.9
	assert new_val - 4.9 < 1e-9 && new_val - 4.9 > -1e-9, 'SGD update expected 4.9, got ${new_val}'
}

fn test_sgd_zeros_grad_after_update() ! {
	c := ctx[f64]()
	mut p := make_var[f64](c, 2.0)
	mut opt := sgd[f64](learning_rate: 0.5)
	opt.params << p
	p.grad.set_nth(0, 4.0)
	opt.update()!
	g := p.grad.get_nth(0)
	assert g == 0.0, 'grad should be zeroed after update, got ${g}'
}

fn test_adam_step_f64_cpu() {
	grad := [1.0, 2.0]
	mut theta := [5.0, 5.0]
	mut m := [0.0, 0.0]
	mut v := [0.0, 0.0]
	adam_step_f64_cpu(grad, mut theta, mut m, mut v, AdamStepParams{
		beta1:   0.9
		beta2:   0.999
		lr_t:    0.001
		epsilon: 1e-8
	})
	assert theta[0] < 5.0
	assert m[0] > 0.0
}

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

fn test_adam_update_moves_param() ! {
	c := ctx[f64]()
	mut p := make_var[f64](c, 1.0)
	mut opt := adam_optimizer[f64](learning_rate: 0.01)
	opt.params << p
	opt.first_moments << vtl.zeros_like[f64](p.grad)
	opt.second_moments << vtl.zeros_like[f64](p.grad)
	p.grad.set_nth(0, 1.0)
	before := p.value.get_nth(0)
	opt.update()!
	after := p.value.get_nth(0)
	assert after < before, 'Adam should decrease param when grad > 0'
}

fn test_adamw_update_moves_param() ! {
	c := ctx[f64]()
	mut p := make_var[f64](c, 1.0)
	mut opt := adamw[f64](learning_rate: 0.01)
	opt.params << p
	opt.first_moments << vtl.zeros_like[f64](p.grad)
	opt.second_moments << vtl.zeros_like[f64](p.grad)
	p.grad.set_nth(0, 1.0)
	before := p.value.get_nth(0)
	opt.update()!
	after := p.value.get_nth(0)
	assert after < before, 'AdamW should decrease param when grad > 0, before=${before} after=${after}'
}

fn test_rmsprop_update_moves_param() ! {
	c := ctx[f64]()
	mut p := make_var[f64](c, 1.0)
	mut opt := rmsprop[f64](learning_rate: 0.01)
	opt.params << p
	opt.sq_avg << vtl.zeros_like[f64](p.grad)
	p.grad.set_nth(0, 1.0)
	before := p.value.get_nth(0)
	opt.update()!
	after := p.value.get_nth(0)
	assert after < before, 'RMSProp should decrease param when grad > 0, before=${before} after=${after}'
}

fn test_adagrad_update_moves_param() ! {
	c := ctx[f64]()
	mut p := make_var[f64](c, 1.0)
	mut opt := adagrad[f64](learning_rate: 0.1)
	opt.params << p
	opt.accumulated_sq_grads << vtl.zeros_like[f64](p.grad)
	p.grad.set_nth(0, 1.0)
	before := p.value.get_nth(0)
	opt.update()!
	after := p.value.get_nth(0)
	assert after < before, 'Adagrad should decrease param when grad > 0, before=${before} after=${after}'
}
