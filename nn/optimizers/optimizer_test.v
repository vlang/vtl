module optimizers

import vtl
import vtl.autograd

fn ctx[T]() &autograd.Context[T] {
	return autograd.ctx[T]()
}

fn make_var[T](c &autograd.Context[T], val f64) &autograd.Variable[T] {
	t := vtl.from_1d([vtl.cast[T](val)]) or { panic(err) }
	return c.variable(t)
}

// SGD: param -= lr * grad
fn test_sgd_update() ! {
	c := ctx[f32]()
	mut p := make_var[f32](c, 5.0)
	mut opt := sgd[f32](learning_rate: 0.1)
	opt.params << p
	p.grad.set_nth(0, 1.0)
	opt.update()!
	new_val := p.value.get_nth(0)
	// 5.0 - 0.1 * 1.0 = 4.9
	assert new_val - 4.9 < 1e-9 && new_val - 4.9 > -1e-9, 'SGD update expected 4.9, got ${new_val}'
}

fn test_sgd_zeros_grad_after_update() ! {
	c := ctx[f32]()
	mut p := make_var[f32](c, 2.0)
	mut opt := sgd[f32](learning_rate: 0.5)
	opt.params << p
	p.grad.set_nth(0, 4.0)
	opt.update()!
	g := p.grad.get_nth(0)
	assert g == 0.0, 'grad should be zeroed after update, got ${g}'
}

fn test_adam_step_f32_cpu_moves_theta() {
	grad := [f32(1.0), f32(2.0)]
	mut theta := [f32(5.0), f32(5.0)]
	mut m := [f32(0.0), f32(0.0)]
	mut v := [f32(0.0), f32(0.0)]
	adam_step_f32_cpu(grad, mut theta, mut m, mut v, AdamStepParams{
		beta1:   0.9
		beta2:   0.999
		lr_t:    0.001
		epsilon: 1e-8
	})
	assert theta[0] < 5.0
	assert m[0] > 0.0
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

fn test_adam_update_moves_param() ! {
	c := ctx[f32]()
	mut p := make_var[f32](c, 1.0)
	mut opt := adam_optimizer[f32](learning_rate: 0.01)
	opt.params << p
	opt.first_moments << vtl.zeros_like[f32](p.grad)
	opt.second_moments << vtl.zeros_like[f32](p.grad)
	p.grad.set_nth(0, 1.0)
	before := p.value.get_nth(0)
	opt.update()!
	after := p.value.get_nth(0)
	assert after < before, 'Adam should decrease param when grad > 0'
}

fn test_adamw_update_moves_param() ! {
	c := ctx[f32]()
	mut p := make_var[f32](c, 1.0)
	mut opt := adamw[f32](learning_rate: 0.01)
	opt.params << p
	opt.first_moments << vtl.zeros_like[f32](p.grad)
	opt.second_moments << vtl.zeros_like[f32](p.grad)
	p.grad.set_nth(0, 1.0)
	before := p.value.get_nth(0)
	opt.update()!
	after := p.value.get_nth(0)
	assert after < before, 'AdamW should decrease param when grad > 0, before=${before} after=${after}'
}

fn test_rmsprop_update_moves_param() ! {
	c := ctx[f32]()
	mut p := make_var[f32](c, 1.0)
	mut opt := rmsprop[f32](learning_rate: 0.01)
	opt.params << p
	opt.sq_avg << vtl.zeros_like[f32](p.grad)
	p.grad.set_nth(0, 1.0)
	before := p.value.get_nth(0)
	opt.update()!
	after := p.value.get_nth(0)
	assert after < before, 'RMSProp should decrease param when grad > 0, before=${before} after=${after}'
}

fn test_adagrad_update_moves_param() ! {
	c := ctx[f32]()
	mut p := make_var[f32](c, 1.0)
	mut opt := adagrad[f32](learning_rate: 0.1)
	opt.params << p
	opt.accumulated_sq_grads << vtl.zeros_like[f32](p.grad)
	p.grad.set_nth(0, 1.0)
	before := p.value.get_nth(0)
	opt.update()!
	after := p.value.get_nth(0)
	assert after < before, 'Adagrad should decrease param when grad > 0, before=${before} after=${after}'
}
