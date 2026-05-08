module optimizers

import math

// Scheduler is the interface for learning rate schedulers.
pub interface Scheduler[T] {
	next_lr(current_lr f64, step int) f64
}

// StepLR decays the learning rate by `gamma` every `step_size` steps.
pub struct StepLR[T] {
	step_size int
	gamma     f64
}

pub fn step_lr[T](step_size int, gamma f64) &StepLR[T] {
	return &StepLR[T]{step_size: step_size, gamma: gamma}
}

pub fn (s &StepLR[T]) next_lr(current_lr f64, step int) f64 {
	return current_lr * math.pow(s.gamma, f64(step / s.step_size))
}

// ExponentialLR decays the learning rate by `gamma` at every step.
pub struct ExponentialLR[T] {
	gamma f64
}

pub fn exponential_lr[T](gamma f64) &ExponentialLR[T] {
	return &ExponentialLR[T]{gamma: gamma}
}

pub fn (s &ExponentialLR[T]) next_lr(current_lr f64, step int) f64 {
	return current_lr * math.pow(s.gamma, f64(step))
}

// CosineAnnealingLR decays using a cosine schedule from `lrd` to 0.
pub struct CosineAnnealingLR[T] {
pub:
	t_max int     // maximum number of iterations
	lrd   f64     // lower bound lr (default: 0)
}

pub fn cosine_annealing_lr[T](t_max int, lrd f64) &CosineAnnealingLR[T] {
	return &CosineAnnealingLR[T]{t_max: t_max, lrd: lrd}
}

pub fn (s &CosineAnnealingLR[T]) next_lr(current_lr f64, step int) f64 {
	if step >= s.t_max {
		return s.lrd
	}
	return s.lrd + (current_lr - s.lrd) * (1.0 + math.cos(math.pi * f64(step) / f64(s.t_max))) / 2.0
}

// ReduceLROnPlateau reduces LR when a metric has stopped improving.
pub struct ReduceLROnPlateau[T] {
	factor       f64
	patience     int
	threshold    f64
	epsilon      f64
	cooldown     int
pub mut:
	wait         int
	current_lr   f64
}

@[params]
pub struct ReduceLROnPlateauConfig {
	factor       f64 = 0.1
	patience     int = 10
	threshold    f64 = 1e-4
	epsilon      f64 = 1e-8
	cooldown     int = 0
}

pub fn reduce_lr_on_plateau[T](config ReduceLROnPlateauConfig) &ReduceLROnPlateau[T] {
	return &ReduceLROnPlateau[T]{
		factor:   config.factor
		patience: config.patience
		threshold: config.threshold
		epsilon:  config.epsilon
		cooldown: config.cooldown
		wait:     0
	}
}

pub fn (mut s ReduceLROnPlateau[T]) next_lr(current_lr f64, step int, metric_delta f64) f64 {
	s.current_lr = current_lr
	if s.wait > 0 {
		s.wait--
		return current_lr
	}
	if metric_delta > s.threshold {
		s.wait = s.patience
		return current_lr
	}
	// Metric improved or within epsilon → reduce LR
	new_lr := current_lr * s.factor
	s.wait = s.patience
	return new_lr
}