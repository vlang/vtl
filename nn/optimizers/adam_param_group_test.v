module optimizers

import vtl
import vtl.autograd

fn test_adam_param_groups_use_their_own_learning_rates() ! {
	c := autograd.ctx[f64]()
	mut fast := c.variable(vtl.from_1d([5.0])!)
	mut slow := c.variable(vtl.from_1d([5.0])!)
	mut opt := adam_optimizer[f64](learning_rate: 0.001)

	fast_group := opt.add_param_group([fast], learning_rate: 0.1)!
	slow_group := opt.add_param_group([slow], learning_rate: 0.01)!
	assert fast_group == 0
	assert slow_group == 1
	assert opt.group_count() == 2
	assert opt.group_learning_rate(slow_group)! == 0.01

	fast.grad.set_nth(0, 1.0)
	slow.grad.set_nth(0, 1.0)
	opt.update()!

	// One Adam step with gradient 1 moves a parameter by about its group LR.
	fast_value := fast.value.get_nth(0)
	slow_value := slow.value.get_nth(0)
	assert fast_value > 4.89 && fast_value < 4.91, 'fast group expected about 4.9, got ${fast_value}'
	assert slow_value > 4.989 && slow_value < 4.991, 'slow group expected about 4.99, got ${slow_value}'
	assert fast.grad.get_nth(0) == 0.0
	assert slow.grad.get_nth(0) == 0.0
}

fn test_adam_param_group_learning_rate_changes_independently() ! {
	c := autograd.ctx[f64]()
	mut p := c.variable(vtl.from_1d([5.0])!)
	mut opt := adam_optimizer[f64](learning_rate: 0.001)
	group := opt.add_param_group([p], learning_rate: 0.1)!

	opt.set_group_learning_rate(group, 0.2)!
	assert opt.group_learning_rate(group)! == 0.2

	p.grad.set_nth(0, 1.0)
	opt.update()!
	value := p.value.get_nth(0)
	assert value > 4.79 && value < 4.81, 'expected about 4.8 after LR change, got ${value}'

	if _ := opt.group_learning_rate(3) {
		assert false, 'out-of-range group index should error'
	}
	if _ := opt.set_group_learning_rate(group, -0.1) {
		assert false, 'negative learning rate should error'
	}
}

fn test_adam_param_group_rejects_invalid_config() ! {
	c := autograd.ctx[f64]()
	mut p := c.variable(vtl.from_1d([5.0])!)
	mut opt := adam_optimizer[f64](learning_rate: 0.001)
	if _ := opt.add_param_group([p], learning_rate: -0.1) {
		assert false, 'negative group learning rate should error'
	}
	if _ := opt.add_param_group([p], beta1: 1.0) {
		assert false, 'beta1 of 1 should error'
	}
	if _ := opt.add_param_group([p], epsilon: 0.0) {
		assert false, 'zero epsilon should error'
	}
	assert opt.group_count() == 0
}
