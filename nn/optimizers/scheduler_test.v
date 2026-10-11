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

fn test_multi_step_lr_decays_at_sorted_milestones() {
	scheduler := multi_step_lr[f64](MultiStepLRConfig{
		milestones: [2, 4, 4]
		gamma:      0.5
	}) or { panic(err) }
	assert scheduler.next_lr(1.0, 0) == 1.0
	assert scheduler.next_lr(1.0, 2) == 0.5
	assert scheduler.next_lr(1.0, 4) == 0.125
}

fn test_multi_step_lr_rejects_unsorted_milestones() {
	assert multi_step_lr_error_for_unsorted_milestones()
}

fn multi_step_lr_error_for_unsorted_milestones() bool {
	_ := multi_step_lr[f64](MultiStepLRConfig{
		milestones: [3, 2]
	}) or {
		return err.msg() == 'milestones must be non-negative and increasing'
	}
	return false
}

fn test_polynomial_lr_decays_to_zero_and_clamps() {
	scheduler := polynomial_lr[f64](PolynomialLRConfig{
		total_iters: 4
		power:       2.0
	}) or { panic(err) }
	assert scheduler.next_lr(0.8, 0) == 0.8
	assert scheduler.next_lr(0.8, 2) == 0.2
	assert scheduler.next_lr(0.8, 4) == 0.0
	assert scheduler.next_lr(0.8, 10) == 0.0
	assert scheduler.next_lr(0.8, -1) == 0.8
}

fn test_constant_lr_scales_until_milestone() {
	scheduler := constant_lr[f64](ConstantLRConfig{
		factor:      0.25
		total_iters: 3
	}) or { panic(err) }
	assert scheduler.next_lr(0.8, 0) == 0.2
	assert scheduler.next_lr(0.8, 2) == 0.2
	assert scheduler.next_lr(0.8, 3) == 0.8
	assert scheduler.next_lr(0.8, 8) == 0.8
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

fn test_cosine_annealing_warm_restarts_equal_cycles() {
	scheduler := cosine_annealing_warm_restarts[f64](CosineAnnealingWarmRestartsConfig{
		t_0:     4
		eta_min: 0.0
	}) or { panic(err) }
	assert math.abs(scheduler.next_lr(1.0, 0) - 1.0) < 1e-12
	assert math.abs(scheduler.next_lr(1.0, 2) - 0.5) < 1e-12
	assert math.abs(scheduler.next_lr(1.0, 3) - 0.1464466094) < 1e-9
	assert math.abs(scheduler.next_lr(1.0, 4) - 1.0) < 1e-12
	assert scheduler.next_lr(1.0, -1) == scheduler.next_lr(1.0, 0)
}

fn test_cosine_annealing_warm_restarts_grows_cycle_lengths() {
	scheduler := cosine_annealing_warm_restarts[f64](CosineAnnealingWarmRestartsConfig{
		t_0:    2
		t_mult: 2
	}) or { panic(err) }
	assert math.abs(scheduler.next_lr(1.0, 1) - 0.5) < 1e-12
	assert math.abs(scheduler.next_lr(1.0, 2) - 1.0) < 1e-12
	assert math.abs(scheduler.next_lr(1.0, 4) - 0.5) < 1e-12
	assert math.abs(scheduler.next_lr(1.0, 6) - 1.0) < 1e-12
}

fn test_cosine_annealing_warm_restarts_rejects_invalid_config() {
	assert cosine_restarts_error_for_zero_period()
}

fn cosine_restarts_error_for_zero_period() bool {
	_ := cosine_annealing_warm_restarts[f64](CosineAnnealingWarmRestartsConfig{
		t_0: 0
	}) or {
		return err.msg() == 't_0 must be positive, t_mult must be at least 1, and eta_min cannot be negative'
	}
	return false
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

fn test_cyclic_lr_triangular_curve_and_default_down_duration() {
	scheduler := cyclic_lr[f64](CyclicLRConfig{
		base_lr:      0.1
		max_lr:       0.5
		step_size_up: 2
	}) or { panic(err) }
	assert scheduler.step_size_up == 2
	assert scheduler.step_size_down == 2
	assert math.abs(scheduler.next_lr(9, 0) - 0.1) < 1e-12
	assert math.abs(scheduler.next_lr(9, 1) - 0.3) < 1e-12
	assert math.abs(scheduler.next_lr(9, 2) - 0.5) < 1e-12
	assert math.abs(scheduler.next_lr(9, 3) - 0.3) < 1e-12
	assert math.abs(scheduler.next_lr(9, 4) - 0.1) < 1e-12
	assert scheduler.next_lr(9, -1) == scheduler.next_lr(9, 0)
}

fn test_cyclic_lr_asymmetric_triangular2_and_exp_range() {
	asymmetric := cyclic_lr[f64](CyclicLRConfig{
		base_lr:        0.1
		max_lr:         0.5
		step_size_up:   2
		step_size_down: 4
	}) or { panic(err) }
	assert math.abs(asymmetric.next_lr(0, 2) - 0.5) < 1e-12
	assert math.abs(asymmetric.next_lr(0, 4) - 0.3) < 1e-12
	assert math.abs(asymmetric.next_lr(0, 6) - 0.1) < 1e-12

	triangular2 := cyclic_lr[f64](CyclicLRConfig{
		base_lr:      0.1
		max_lr:       0.5
		step_size_up: 2
		mode:         .triangular2
	}) or { panic(err) }
	assert math.abs(triangular2.next_lr(0, 6) - 0.3) < 1e-12

	exp_range := cyclic_lr[f64](CyclicLRConfig{
		base_lr:      0.1
		max_lr:       0.5
		step_size_up: 2
		mode:         .exp_range
		gamma:        0.5
	}) or { panic(err) }
	assert math.abs(exp_range.next_lr(0, 2) - 0.2) < 1e-12
}

fn test_cyclic_lr_rejects_invalid_parameters() {
	assert cyclic_lr_error_for_invalid_up_duration()
}

fn cyclic_lr_error_for_invalid_up_duration() bool {
	_ := cyclic_lr[f64](CyclicLRConfig{
		base_lr:      0.1
		max_lr:       0.5
		step_size_up: 0
	}) or {
		return err.msg() == 'step_size_up must be positive and step_size_down cannot be negative'
	}
	return false
}
