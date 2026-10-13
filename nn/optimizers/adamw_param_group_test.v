module optimizers

import vtl
import vtl.autograd

fn test_adamw_param_groups_use_their_own_learning_rates() ! {
	c := autograd.ctx[f64]()
	mut fast := c.variable(vtl.from_1d([5.0])!)
	mut slow := c.variable(vtl.from_1d([5.0])!)
	mut opt := adamw[f64](learning_rate: 0.001)

	fast_group := opt.add_param_group([fast], learning_rate: 0.1, weight_decay: 0.0)!
	slow_group := opt.add_param_group([slow], learning_rate: 0.01, weight_decay: 0.0)!
	assert fast_group == 0
	assert slow_group == 1
	assert opt.group_count() == 2
	assert opt.group_learning_rate(fast_group)! == 0.1

	fast.grad.set_nth(0, 1.0)
	slow.grad.set_nth(0, 1.0)
	opt.update()!

	fast_value := fast.value.get_nth(0)
	slow_value := slow.value.get_nth(0)
	assert fast_value > 4.89 && fast_value < 4.91, 'fast group expected about 4.9, got ${fast_value}'
	assert slow_value > 4.989 && slow_value < 4.991, 'slow group expected about 4.99, got ${slow_value}'
	assert fast.grad.get_nth(0) == 0.0
	assert slow.grad.get_nth(0) == 0.0
}

fn test_adamw_param_group_weight_decay_is_per_group() ! {
	c := autograd.ctx[f64]()
	mut decayed := c.variable(vtl.from_1d([5.0])!)
	mut plain := c.variable(vtl.from_1d([5.0])!)
	mut opt := adamw[f64](learning_rate: 0.001)

	// Zero gradient: only decoupled weight decay moves the decayed parameter.
	opt.add_param_group([decayed], learning_rate: 0.1, weight_decay: 0.5)!
	opt.add_param_group([plain], learning_rate: 0.1, weight_decay: 0.0)!

	decayed.grad.set_nth(0, 0.0)
	plain.grad.set_nth(0, 0.0)
	opt.update()!

	assert decayed.value.get_nth(0) < 4.99
	assert plain.value.get_nth(0) == 5.0
}

fn test_adamw_param_group_rejects_invalid_config() ! {
	c := autograd.ctx[f64]()
	mut p := c.variable(vtl.from_1d([5.0])!)
	mut opt := adamw[f64](learning_rate: 0.001)
	if _ := opt.add_param_group([p], weight_decay: -0.1) {
		assert false, 'negative weight decay should error'
	}
	if _ := opt.add_param_group([p], beta2: 1.0) {
		assert false, 'beta2 of 1 should error'
	}
	if _ := opt.set_group_learning_rate(0, 0.1) {
		assert false, 'missing group should error'
	}
	assert opt.group_count() == 0
}
