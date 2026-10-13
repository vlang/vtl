module optimizers

import vtl
import vtl.autograd

fn test_sgd_param_groups_use_their_own_learning_rates() ! {
	c := autograd.ctx[f64]()
	mut fast := c.variable(vtl.from_1d([5.0])!)
	mut slow := c.variable(vtl.from_1d([5.0])!)
	mut opt := sgd[f64](learning_rate: 0.01)

	fast_group := opt.add_param_group([fast], learning_rate: 0.5)!
	slow_group := opt.add_param_group([slow], learning_rate: 0.1)!
	assert fast_group == 0
	assert slow_group == 1
	assert opt.group_count() == 2
	assert opt.group_learning_rate(fast_group)! == 0.5

	fast.grad.set_nth(0, 1.0)
	slow.grad.set_nth(0, 1.0)
	opt.update()!

	assert fast.value.get_nth(0) == 4.5 // 5 - 0.5 * 1
	assert slow.value.get_nth(0) == 4.9 // 5 - 0.1 * 1
	// grouped update still zeros gradients
	assert fast.grad.get_nth(0) == 0.0
	assert slow.grad.get_nth(0) == 0.0
}

fn test_sgd_param_group_learning_rate_can_change_independently() ! {
	c := autograd.ctx[f64]()
	mut p := c.variable(vtl.from_1d([5.0])!)
	mut opt := sgd[f64](learning_rate: 0.01)
	group := opt.add_param_group([p], learning_rate: 0.5)!

	opt.set_group_learning_rate(group, 0.25)!
	assert opt.group_learning_rate(group)! == 0.25

	p.grad.set_nth(0, 2.0)
	opt.update()!
	assert p.value.get_nth(0) == 4.5 // 5 - 0.25 * 2

	if _ := opt.group_learning_rate(7) {
		assert false, 'out-of-range group index should error'
	}
	if _ := opt.set_group_learning_rate(-1, 0.1) {
		assert false, 'negative group index should error'
	}
	if _ := opt.set_group_learning_rate(group, -0.1) {
		assert false, 'negative learning rate should error'
	}
}

fn test_sgd_param_groups_keep_independent_momentum_buffers() ! {
	c := autograd.ctx[f64]()
	mut a := c.variable(vtl.from_1d([5.0])!)
	mut b := c.variable(vtl.from_1d([5.0])!)
	mut opt := sgd[f64](learning_rate: 0.01)
	opt.add_param_group([a], learning_rate: 0.1, momentum: 0.9)!
	opt.add_param_group([b], learning_rate: 0.1)!

	// first step: momentum group initializes its buffer from the gradient
	a.grad.set_nth(0, 1.0)
	b.grad.set_nth(0, 1.0)
	opt.update()!
	assert a.value.get_nth(0) == 4.9
	assert b.value.get_nth(0) == 4.9

	// second step: only the momentum group carries the previous gradient
	a.grad.set_nth(0, 1.0)
	b.grad.set_nth(0, 1.0)
	opt.update()!
	// buffer = 0.9 * 1 + 1 = 1.9; step = 0.1 * 1.9 = 0.19
	assert a.value.get_nth(0) > 4.9 - 0.19 - 1e-12 && a.value.get_nth(0) < 4.9 - 0.19 + 1e-12
	assert b.value.get_nth(0) > 4.8 - 1e-12 && b.value.get_nth(0) < 4.8 + 1e-12
}

fn test_sgd_param_group_rejects_invalid_config() ! {
	c := autograd.ctx[f64]()
	mut p := c.variable(vtl.from_1d([5.0])!)
	mut opt := sgd[f64](learning_rate: 0.01)
	if _ := opt.add_param_group([p], learning_rate: -1.0) {
		assert false, 'negative group learning rate should error'
	}
	if _ := opt.add_param_group([p], learning_rate: 0.1, nesterov: true) {
		assert false, 'nesterov without momentum should error'
	}
	assert opt.group_count() == 0
}
