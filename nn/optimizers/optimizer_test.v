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
