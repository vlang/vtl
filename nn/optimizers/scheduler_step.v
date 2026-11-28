module optimizers

import math

// step_group_learning_rates advances a scheduler across optimizer parameter
// groups. `current_lrs` is indexed by group; `next_lr` computes a group's new
// rate from its current rate and the training step; `apply` writes a rate to a
// group.
//
// Every rate is computed and validated before `apply` is called, so an
// invalid scheduled rate cannot leave the optimizer partially updated. The
// caller-supplied callback keeps this helper independent of the concrete
// optimizer type while still giving each group its own base rate, matching
// PyTorch's per-group scheduler behavior.
pub fn step_group_learning_rates(current_lrs []f64, step int, next_lr fn (f64, int) f64, apply fn (int, f64) !) ![]f64 {
	if current_lrs.len == 0 {
		return error('scheduler step requires at least one parameter group')
	}
	mut updated := []f64{len: current_lrs.len}
	for i, lr in current_lrs {
		if !math.is_finite(lr) || lr < 0.0 {
			return error('invalid current learning rate ${lr} for group ${i}')
		}
		next := next_lr(lr, step)
		if !math.is_finite(next) || next < 0.0 {
			return error('invalid scheduled learning rate ${next} for group ${i}')
		}
		updated[i] = next
	}
	for i, lr in updated {
		apply(i, lr) or {
			return error('failed to apply learning rate to group ${i}: ${err.msg()}')
		}
	}
	return updated
}
