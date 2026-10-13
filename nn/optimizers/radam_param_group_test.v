module optimizers

import math
import vtl
import vtl.autograd

fn test_radam_param_groups_use_their_own_learning_rates() ! {
	c := autograd.ctx[f64]()
	mut fast := c.variable(vtl.from_1d([5.0])!)
	mut slow := c.variable(vtl.from_1d([5.0])!)
	mut opt := radam_optimizer[f64](learning_rate: 0.001)

	fast_group := opt.add_param_group([fast], learning_rate: 0.1)!
	slow_group := opt.add_param_group([slow], learning_rate: 0.01)!
	assert fast_group == 0
	assert slow_group == 1
	assert opt.group_count() == 2

	fast.grad.set_nth(0, 1.0)
	slow.grad.set_nth(0, 1.0)
	opt.update()!

	// First RAdam step is unadaptive (rho_t = 1 <= 5): m = 0.1 and
	// m_hat = 1, so theta moves by exactly the group learning rate.
	assert math.abs(fast.value.get_nth(0) - 4.9) < 1e-9, 'fast ${fast.value.get_nth(0)}'
	assert math.abs(slow.value.get_nth(0) - 4.99) < 1e-9, 'slow ${slow.value.get_nth(0)}'
	assert fast.grad.get_nth(0) == 0.0
	assert slow.grad.get_nth(0) == 0.0
	assert opt.groups[fast_group].step == 1
	assert opt.groups[slow_group].step == 1
}

fn test_radam_param_group_rejects_invalid_config() ! {
	c := autograd.ctx[f64]()
	mut p := c.variable(vtl.from_1d([5.0])!)
	mut opt := radam_optimizer[f64](learning_rate: 0.001)
	if _ := opt.add_param_group([p], learning_rate: -0.1) {
		assert false, 'negative learning rate should error'
	}
	if _ := opt.add_param_group([p], epsilon: 0.0) {
		assert false, 'zero epsilon should error'
	}
	if _ := opt.set_group_learning_rate(0, 0.1) {
		assert false, 'missing group should error'
	}
	assert opt.group_count() == 0
}
