module optimizers

import math

// Scheduler is the interface for learning rate schedulers.
// Scheduler is the interface for learning rate schedulers.
// Implementations update the learning rate based on the current training step.
pub interface Scheduler[T] {
	next_lr(current_lr f64, step int) f64
}

// StepLR decays the learning rate by `gamma` every `step_size` steps.
pub struct StepLR[T] {
	step_size int
	gamma     f64
}

// step_lr creates a StepLR scheduler: LR decays by `gamma` every `step_size` steps.
// e.g. step_size=30, gamma=0.1 → LR is 10× smaller after each 30 steps.
pub fn step_lr[T](step_size int, gamma f64) &StepLR[T] {
	return &StepLR[T]{
		step_size: step_size
		gamma:     gamma
	}
}

// next_lr exposes this operation as part of the public API.
pub fn (s &StepLR[T]) next_lr(current_lr f64, step int) f64 {
	return current_lr * math.pow(s.gamma, f64(step / s.step_size))
}

// ExponentialLR decays the learning rate by `gamma` at every step.
pub struct ExponentialLR[T] {
	gamma f64
}

// exponential_lr creates an ExponentialLR scheduler: LR decays by `gamma` at every step.
// new_lr = current_lr * gamma^step
pub fn exponential_lr[T](gamma f64) &ExponentialLR[T] {
	return &ExponentialLR[T]{
		gamma: gamma
	}
}

// next_lr exposes this operation as part of the public API.
pub fn (s &ExponentialLR[T]) next_lr(current_lr f64, step int) f64 {
	return current_lr * math.pow(s.gamma, f64(step))
}

// CosineAnnealingLR decays using a cosine schedule from `lrd` to 0.
pub struct CosineAnnealingLR[T] {
pub:
	t_max int // maximum number of iterations
	lrd   f64 // lower bound lr (default: 0)
}

// cosine_annealing_lr creates a CosineAnnealingLR scheduler: LR decays from `current_lr` to `lrd`
// following a cosine schedule over `t_max` steps.
pub fn cosine_annealing_lr[T](t_max int, lrd f64) &CosineAnnealingLR[T] {
	return &CosineAnnealingLR[T]{
		t_max: t_max
		lrd:   lrd
	}
}

// next_lr exposes this operation as part of the public API.
pub fn (s &CosineAnnealingLR[T]) next_lr(current_lr f64, step int) f64 {
	if step >= s.t_max {
		return s.lrd
	}
	return s.lrd + (current_lr - s.lrd) * (1.0 + math.cos(math.pi * f64(step) / f64(s.t_max))) / 2.0
}

// LinearLR scales the base learning rate linearly between two factors.
pub struct LinearLR[T] {
	start_factor f64
	end_factor   f64
	total_iters  int
}

// linear_lr creates a linear learning-rate schedule. The returned value is
// based on `base_lr` and interpolates from `start_factor` to `end_factor`.
pub fn linear_lr[T](start_factor f64, end_factor f64, total_iters int) &LinearLR[T] {
	return &LinearLR[T]{
		start_factor: start_factor
		end_factor:   end_factor
		total_iters:  total_iters
	}
}

// next_lr exposes this operation as part of the public API.
pub fn (s &LinearLR[T]) next_lr(base_lr f64, step int) f64 {
	if s.total_iters <= 0 {
		return base_lr * s.end_factor
	}
	clamped_step := math.min(math.max(step, 0), s.total_iters)
	ratio := f64(clamped_step) / f64(s.total_iters)
	factor := s.start_factor + ratio * (s.end_factor - s.start_factor)
	return base_lr * factor
}

// ReduceLROnPlateau reduces LR when a metric has stopped improving.
pub struct ReduceLROnPlateau[T] {
	factor    f64
	patience  int
	threshold f64
	epsilon   f64
	cooldown  int
mut:
	cooldown_remaining int
pub mut:
	wait       int
	current_lr f64
}

// ReduceLROnPlateauConfig defines a public data structure for this module.

// ReduceLROnPlateauConfig defines a public data structure for this module.
@[params]
pub struct ReduceLROnPlateauConfig {
pub:
	factor    f64 = 0.1
	patience  int = 10
	threshold f64 = 1e-4
	epsilon   f64 = 1e-8
	cooldown  int
}

// reduce_lr_on_plateau creates a ReduceLROnPlateau scheduler.
// Reduces LR by `factor` when the monitored metric stops improving for `patience` steps.
// Pass `metric_delta` to `next_lr` each step: a negative delta below
// `-threshold` means improvement; other finite deltas count toward patience.
pub fn reduce_lr_on_plateau[T](config ReduceLROnPlateauConfig) &ReduceLROnPlateau[T] {
	return &ReduceLROnPlateau[T]{
		factor:             config.factor
		patience:           config.patience
		threshold:          config.threshold
		epsilon:            config.epsilon
		cooldown:           config.cooldown
		wait:               0
		cooldown_remaining: 0
	}
}

// next_lr exposes this operation as part of the public API.
pub fn (mut s ReduceLROnPlateau[T]) next_lr(current_lr f64, _ int, metric_delta f64) f64 {
	s.current_lr = current_lr
	if math.is_nan(metric_delta) {
		return current_lr
	}
	if s.cooldown_remaining > 0 {
		s.cooldown_remaining--
		s.wait = 0
		return current_lr
	}
	// A sufficiently negative delta means the monitored metric improved.
	if metric_delta < -s.threshold {
		s.wait = 0
		return current_lr
	}
	// Deltas near zero or above indicate no significant improvement.
	s.wait++
	if s.wait <= s.patience {
		return current_lr
	}
	s.wait = 0
	new_lr := current_lr * s.factor
	if current_lr - new_lr <= s.epsilon {
		return current_lr
	}
	s.cooldown_remaining = s.cooldown
	return new_lr
}
