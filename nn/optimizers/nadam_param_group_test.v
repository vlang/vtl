module optimizers

import math
import vtl
import vtl.autograd

fn test_nadam_param_groups_use_their_own_learning_rates() ! {
	c := autograd.ctx[f64]()
	mut fast := c.variable(vtl.from_1d([5.0])!)
	mut slow := c.variable(vtl.from_1d([5.0])!)
	mut opt := nadam_optimizer[f64](learning_rate: 0.002)

	fast_group := opt.add_param_group([fast], learning_rate: 0.1)!
	slow_group := opt.add_param_group([slow], learning_rate: 0.01)!
	assert fast_group == 0
	assert slow_group == 1
	assert opt.group_count() == 2
	assert opt.group_learning_rate(slow_group)! == 0.01

	fast.grad.set_nth(0, 1.0)
	slow.grad.set_nth(0, 1.0)
	opt.update()!

	// Dozat NAdam first step from theta=5, grad=1, default betas/decay:
	// m_hat combines the Nesterov lookahead terms, v_hat = 1 here.
	beta1 := 0.9
	decay := 0.004
	mu_t := beta1 * (1.0 - 0.5 * math.pow(0.96, decay))
	mu_next := beta1 * (1.0 - 0.5 * math.pow(0.96, 2.0 * decay))
	m_hat := mu_next * 0.1 / (1.0 - mu_t * mu_next) + (1.0 - mu_t) / (1.0 - mu_t)
	fast_expected := 5.0 - 0.1 * m_hat
	slow_expected := 5.0 - 0.01 * m_hat
	assert math.abs(fast.value.get_nth(0) - fast_expected) < 1e-6, 'fast ${fast.value.get_nth(0)} vs ${fast_expected}'
	assert math.abs(slow.value.get_nth(0) - slow_expected) < 1e-6, 'slow ${slow.value.get_nth(0)} vs ${slow_expected}'
	assert fast.grad.get_nth(0) == 0.0
	assert slow.grad.get_nth(0) == 0.0
	// Each group keeps its own step counter and momentum product.
	assert opt.groups[fast_group].step == 1
	assert opt.groups[slow_group].step == 1
	assert opt.step == 1
}

fn test_nadam_param_group_rejects_invalid_config() ! {
	c := autograd.ctx[f64]()
	mut p := c.variable(vtl.from_1d([5.0])!)
	mut opt := nadam_optimizer[f64](learning_rate: 0.002)
	if _ := opt.add_param_group([p], beta1: 1.0) {
		assert false, 'beta1 of 1 should error'
	}
	if _ := opt.add_param_group([p], momentum_decay: -0.1) {
		assert false, 'negative momentum decay should error'
	}
	if _ := opt.set_group_learning_rate(0, 0.1) {
		assert false, 'missing group should error'
	}
	assert opt.group_count() == 0
}
