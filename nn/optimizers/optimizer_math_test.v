module optimizers

import math
import vtl
import vtl.autograd

// Regression tests with exact first-step values. napply passes the state
// tensor itself as vals[0] and the gradient as vals[1]; AdamW and AdaGrad
// previously read them swapped, computing m = beta*grad + (1-beta)*m_old
// and G = grad + G_old^2 instead of the published updates.

fn test_adamw_flat_first_step_matches_reference() ! {
	c := autograd.ctx[f64]()
	mut p := c.variable(vtl.from_1d([5.0])!)
	mut opt := adamw[f64](learning_rate: 0.1, weight_decay: 0.0)
	opt.params << p
	opt.first_moments << vtl.zeros_like[f64](p.grad)
	opt.second_moments << vtl.zeros_like[f64](p.grad)

	p.grad.set_nth(0, 1.0)
	opt.update()!

	// m = 0.1, v = 0.001, lr_t = 0.1 * sqrt(1-0.999) / (1-0.9):
	// theta = 5 - lr_t * m / sqrt(v) = 5 - 0.1 = 4.9.
	value := p.value.get_nth(0)
	assert math.abs(value - 4.9) < 1e-6, 'AdamW first step expected 4.9, got ${value}'
	assert math.abs(opt.first_moments[0].get_nth(0) - 0.1) < 1e-12
	assert math.abs(opt.second_moments[0].get_nth(0) - 0.001) < 1e-12
}

fn test_adagrad_first_step_matches_reference() ! {
	c := autograd.ctx[f64]()
	mut p := c.variable(vtl.from_1d([5.0])!)
	mut opt := adagrad[f64](learning_rate: 0.1)
	opt.params << p
	opt.accumulated_sq_grads << vtl.zeros_like[f64](p.grad)

	p.grad.set_nth(0, 2.0)
	opt.update()!

	// G = 4, theta = 5 - 0.1 * 2 / sqrt(4) = 4.9.
	value := p.value.get_nth(0)
	assert math.abs(value - 4.9) < 1e-6, 'AdaGrad first step expected 4.9, got ${value}'
	assert math.abs(opt.accumulated_sq_grads[0].get_nth(0) - 4.0) < 1e-12
}
