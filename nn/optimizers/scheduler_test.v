module optimizers

import math

fn test_reduce_lr_on_plateau_waits_for_patience_and_ignores_improvements() {
	mut scheduler := reduce_lr_on_plateau[f64](ReduceLROnPlateauConfig{
		factor:    0.5
		patience:  2
		threshold: 0.01
		epsilon:   0
	})
	mut lr := 1.0
	lr = scheduler.next_lr(lr, 0, -0.2)
	assert lr == 1.0
	assert scheduler.wait == 0
	lr = scheduler.next_lr(lr, 1, 0.2)
	assert lr == 1.0
	assert scheduler.wait == 1
	lr = scheduler.next_lr(lr, 2, -0.2)
	assert lr == 1.0
	assert scheduler.wait == 0
	lr = scheduler.next_lr(lr, 3, 0)
	assert lr == 1.0
	lr = scheduler.next_lr(lr, 4, 0.2)
	assert lr == 1.0
	lr = scheduler.next_lr(lr, 5, 0.2)
	assert math.abs(lr - 0.5) < 1e-12
	assert scheduler.wait == 0
}

fn test_reduce_lr_on_plateau_honors_cooldown() {
	mut scheduler := reduce_lr_on_plateau[f64](ReduceLROnPlateauConfig{
		factor:   0.5
		patience: 0
		cooldown: 2
		epsilon:  0
	})
	mut lr := scheduler.next_lr(1.0, 0, 1.0)
	assert lr == 0.5
	lr = scheduler.next_lr(lr, 1, 1.0)
	assert lr == 0.5
	lr = scheduler.next_lr(lr, 2, 1.0)
	assert lr == 0.5
	lr = scheduler.next_lr(lr, 3, 1.0)
	assert lr == 0.25
}

fn test_reduce_lr_on_plateau_respects_epsilon_and_nan_delta() {
	mut scheduler := reduce_lr_on_plateau[f64](ReduceLROnPlateauConfig{
		factor:   0.5
		patience: 0
	})
	assert scheduler.next_lr(1e-9, 0, 1.0) == 1e-9
	assert scheduler.next_lr(1e-9, 1, math.nan()) == 1e-9
}

fn test_linear_lr_interpolates_and_saturates() {
	scheduler := linear_lr[f64](0.25, 1.0, 4)
	assert scheduler.next_lr(0.8, 0) == 0.2
	assert math.abs(scheduler.next_lr(0.8, 2) - 0.5) < 1e-12
	assert scheduler.next_lr(0.8, 4) == 0.8
	assert scheduler.next_lr(0.8, 10) == 0.8
	assert scheduler.next_lr(0.8, -1) == 0.2
}

fn test_linear_lr_zero_or_negative_duration_uses_end_factor() {
	assert math.abs(linear_lr[f64](0.25, 0.75, 0).next_lr(0.8, 0) - 0.6) < 1e-12
	assert math.abs(linear_lr[f64](0.25, 0.75, -2).next_lr(0.8, 0) - 0.6) < 1e-12
}

fn test_one_cycle_lr_two_phase_cosine() {
	scheduler := one_cycle_lr[f64](OneCycleLRConfig{
		max_lr:      1.0
		total_steps: 10
	}) or { panic(err) }
	assert math.abs(scheduler.next_lr(9.0, 0) - 0.04) < 1e-12
	assert math.abs(scheduler.next_lr(9.0, 2) - 1.0) < 1e-12
	assert math.abs(scheduler.next_lr(9.0, 3) - 0.9504846) < 1e-6
	assert math.abs(scheduler.next_lr(9.0, 9) - 0.000004) < 1e-12
	assert scheduler.next_lr(9.0, 20) == scheduler.next_lr(9.0, 9)
	assert scheduler.next_lr(9.0, -1) == scheduler.next_lr(9.0, 0)
}

fn test_one_cycle_lr_linear_and_three_phase() {
	linear := one_cycle_lr[f64](OneCycleLRConfig{
		max_lr:          1.0
		total_steps:     10
		anneal_strategy: .linear
	}) or { panic(err) }
	assert math.abs(linear.next_lr(0, 3) - (1.0 - 0.999996 / 7.0)) < 1e-12

	three_phase := one_cycle_lr[f64](OneCycleLRConfig{
		max_lr:      1.0
		total_steps: 10
		three_phase: true
	}) or { panic(err) }
	assert math.abs(three_phase.next_lr(0, 2) - 1.0) < 1e-12
	assert math.abs(three_phase.next_lr(0, 4) - 0.04) < 1e-12
	assert math.abs(three_phase.next_lr(0, 9) - 0.000004) < 1e-12
}

fn test_one_cycle_lr_degenerate_cycle_uses_final_rate() {
	scheduler := one_cycle_lr[f64](OneCycleLRConfig{
		max_lr:      1.0
		total_steps: 1
	}) or { panic(err) }
	assert math.abs(scheduler.next_lr(0, 0) - 0.000004) < 1e-12
}

fn test_one_cycle_lr_rejects_invalid_parameters() {
	assert one_cycle_lr_error_for_invalid_steps()
}

fn one_cycle_lr_error_for_invalid_steps() bool {
	_ := one_cycle_lr[f64](OneCycleLRConfig{
		max_lr:      1.0
		total_steps: 0
	}) or {
		return err.msg() == 'total_steps must be positive'
	}
	return false
}
