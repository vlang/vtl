module optimizers

import vtl
import vtl.autograd

fn test_step_group_learning_rates_uses_each_groups_base_rate() ! {
	scheduler := step_lr[f64](2, 0.5)
	mut applied := []f64{len: 2}
	updated := step_group_learning_rates([0.1, 0.01], 3,
		fn [scheduler] (lr f64, step int) f64 {
			return scheduler.next_lr(lr, step)
		},
		fn [mut applied] (group int, lr f64) ! {
			applied[group] = lr
		})!
	assert updated == [0.05, 0.005]
	assert applied == [0.05, 0.005]
}

fn test_step_group_learning_rates_rejects_empty_group_list() ! {
	step_group_learning_rates([]f64{}, 0,
		fn (lr f64, step int) f64 {
			return lr
		},
		fn (group int, lr f64) ! {}) or {
		assert err.msg().contains('at least one parameter group')
		return
	}
	assert false, 'an empty group list should be rejected'
}

fn test_step_group_learning_rates_rejects_invalid_current_rate() ! {
	mut calls := 0
	step_group_learning_rates([0.1, -0.1], 0,
		fn (lr f64, step int) f64 {
			return lr
		},
		fn [mut calls] (group int, lr f64) ! {
			calls++
		}) or {
		assert err.msg().contains('invalid current learning rate')
		assert calls == 0
		return
	}
	assert false, 'a negative current rate should be rejected'
}

fn test_step_group_learning_rates_rejects_invalid_scheduled_rate() ! {
	mut calls := 0
	step_group_learning_rates([0.1, 0.01], 0,
		fn (lr f64, step int) f64 {
			return -1.0
		},
		fn [mut calls] (group int, lr f64) ! {
			calls++
		}) or {
		assert err.msg().contains('invalid scheduled learning rate')
		assert calls == 0
		return
	}
	assert false, 'a negative scheduled rate should be rejected'
}

fn test_step_group_learning_rates_propagates_apply_errors() ! {
	step_group_learning_rates([0.1, 0.01], 0,
		fn (lr f64, step int) f64 {
			return lr
		},
		fn (group int, lr f64) ! {
			if group == 1 {
				return error('group 1 is read-only')
			}
		}) or {
		assert err.msg().contains('failed to apply learning rate to group 1')
		assert err.msg().contains('group 1 is read-only')
		return
	}
	assert false, 'apply errors should be propagated'
}

fn test_step_group_learning_rates_updates_adam_groups() ! {
	c := autograd.ctx[f64]()
	p0 := c.variable(vtl.from_1d([1.0])!)
	p1 := c.variable(vtl.from_1d([1.0])!)
	mut opt := adam_optimizer[f64](learning_rate: 0.001)
	fast_group := opt.add_param_group([p0], learning_rate: 0.1)!
	slow_group := opt.add_param_group([p1], learning_rate: 0.01)!
	scheduler := step_lr[f64](2, 0.5)

	step_group_learning_rates([
		opt.group_learning_rate(fast_group)!
		opt.group_learning_rate(slow_group)!,
	], 2,
		fn [scheduler] (lr f64, step int) f64 {
			return scheduler.next_lr(lr, step)
		},
		fn [mut opt] (group int, lr f64) ! {
			opt.set_group_learning_rate(group, lr)!
		})!

	assert opt.group_learning_rate(fast_group)! == 0.05
	assert opt.group_learning_rate(slow_group)! == 0.005
}
