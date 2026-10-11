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
