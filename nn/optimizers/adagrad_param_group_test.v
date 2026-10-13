module optimizers

import math
import vtl
import vtl.autograd

fn test_adagrad_param_groups_use_their_own_learning_rates() ! {
	c := autograd.ctx[f64]()
	mut fast := c.variable(vtl.from_1d([5.0])!)
	mut slow := c.variable(vtl.from_1d([5.0])!)
	mut opt := adagrad[f64](learning_rate: 0.01)

	fast_group := opt.add_param_group([fast], learning_rate: 0.1)!
	slow_group := opt.add_param_group([slow], learning_rate: 0.01)!
	assert fast_group == 0
	assert slow_group == 1
	assert opt.group_count() == 2
	assert opt.group_learning_rate(fast_group)! == 0.1

	fast.grad.set_nth(0, 2.0)
	slow.grad.set_nth(0, 2.0)
	opt.update()!

	// G = 4 for both: fast 5 - 0.1*2/2 = 4.9, slow 5 - 0.01*2/2 = 4.99.
	assert math.abs(fast.value.get_nth(0) - 4.9) < 1e-6
	assert math.abs(slow.value.get_nth(0) - 4.99) < 1e-6
	assert fast.grad.get_nth(0) == 0.0
	assert slow.grad.get_nth(0) == 0.0

	// Accumulators are independent per group and keep accumulating.
	fast.grad.set_nth(0, 2.0)
	slow.grad.set_nth(0, 0.0)
	opt.update()!
	assert math.abs(opt.groups[fast_group].accumulated_sq_grads[0].get_nth(0) - 8.0) < 1e-12
	assert math.abs(opt.groups[slow_group].accumulated_sq_grads[0].get_nth(0) - 4.0) < 1e-12
}

fn test_adagrad_param_group_rejects_invalid_config() ! {
	c := autograd.ctx[f64]()
	mut p := c.variable(vtl.from_1d([5.0])!)
	mut opt := adagrad[f64](learning_rate: 0.01)
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
