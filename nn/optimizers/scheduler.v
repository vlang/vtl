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

// MultiStepLRConfig defines the epochs at which the rate is decayed.
@[params]
pub struct MultiStepLRConfig {
pub:
	milestones []int
	gamma      f64 = 0.1
}

// MultiStepLR decays the base learning rate at each configured milestone.
pub struct MultiStepLR[T] {
	milestones []int
	gamma      f64
}

// multi_step_lr creates a scheduler whose milestones must be in increasing order.
// Repeated milestones apply repeated decay at the same step, like PyTorch.
pub fn multi_step_lr[T](config MultiStepLRConfig) !&MultiStepLR[T] {
	if config.gamma <= 0.0 {
		return error('gamma must be positive')
	}
	for i, milestone in config.milestones {
		if milestone < 0 || (i > 0 && milestone < config.milestones[i - 1]) {
			return error('milestones must be non-negative and increasing')
		}
	}
	return &MultiStepLR[T]{
		milestones: config.milestones.clone()
		gamma:      config.gamma
	}
}

// next_lr returns the rate for a zero-based epoch, based on the base rate.
pub fn (s &MultiStepLR[T]) next_lr(base_lr f64, step int) f64 {
	mut decays := 0
	for milestone in s.milestones {
		if step >= milestone {
			decays++
		}
	}
	return base_lr * math.pow(s.gamma, f64(decays))
}

// PolynomialLRConfig defines the duration and power of a polynomial decay.
@[params]
pub struct PolynomialLRConfig {
pub:
	total_iters int = 5
	power       f64 = 1.0
}

// PolynomialLR decays the base learning rate to zero over total_iters steps.
pub struct PolynomialLR[T] {
	total_iters int
	power       f64
}

// polynomial_lr creates a polynomial decay scheduler.
pub fn polynomial_lr[T](config PolynomialLRConfig) !&PolynomialLR[T] {
	if config.total_iters <= 0 || config.power < 0.0 {
		return error('total_iters must be positive and power must be non-negative')
	}
	return &PolynomialLR[T]{
		total_iters: config.total_iters
		power:       config.power
	}
}

// next_lr returns the polynomially decayed rate for a zero-based step.
pub fn (s &PolynomialLR[T]) next_lr(base_lr f64, step int) f64 {
	clamped_step := math.min(math.max(step, 0), s.total_iters)
	factor := math.pow(1.0 - f64(clamped_step) / f64(s.total_iters), s.power)
	return base_lr * factor
}

// ConstantLRConfig defines a temporary constant scaling of the base rate.
@[params]
pub struct ConstantLRConfig {
pub:
	factor      f64 = 1.0 / 3.0
	total_iters int = 5
}

// ConstantLR scales the base rate by factor for a fixed number of steps.
pub struct ConstantLR[T] {
	factor      f64
	total_iters int
}

// constant_lr creates a temporary constant-rate scaling scheduler.
pub fn constant_lr[T](config ConstantLRConfig) !&ConstantLR[T] {
	if config.factor < 0.0 || config.total_iters < 0 {
		return error('factor and total_iters must be non-negative')
	}
	return &ConstantLR[T]{
		factor:      config.factor
		total_iters: config.total_iters
	}
}

// next_lr returns the scaled rate before total_iters and the base rate afterward.
pub fn (s &ConstantLR[T]) next_lr(base_lr f64, step int) f64 {
	if step < s.total_iters {
		return base_lr * s.factor
	}
	return base_lr
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

// CosineAnnealingWarmRestartsConfig defines a cosine schedule with periodic restarts.
@[params]
pub struct CosineAnnealingWarmRestartsConfig {
pub:
	t_0     int
	t_mult  int = 1
	eta_min f64
}

// CosineAnnealingWarmRestarts restarts the learning rate at the start of each
// cycle and can increase cycle lengths geometrically with t_mult.
pub struct CosineAnnealingWarmRestarts[T] {
	t_0     int
	t_mult  int
	eta_min f64
}

// cosine_annealing_warm_restarts creates the scheduler. `eta_min` should not
// exceed the base learning rate passed to next_lr.
pub fn cosine_annealing_warm_restarts[T](config CosineAnnealingWarmRestartsConfig) !&CosineAnnealingWarmRestarts[T] {
	if config.t_0 <= 0 || config.t_mult < 1 || config.eta_min < 0.0 {
		return error('t_0 must be positive, t_mult must be at least 1, and eta_min cannot be negative')
	}
	return &CosineAnnealingWarmRestarts[T]{
		t_0:     config.t_0
		t_mult:  config.t_mult
		eta_min: config.eta_min
	}
}

// next_lr returns the rate for a zero-based step, with each cycle beginning at
// the base rate and annealing toward eta_min.
pub fn (s &CosineAnnealingWarmRestarts[T]) next_lr(base_lr f64, step int) f64 {
	clamped_step := if step < 0 { 0 } else { step }
	mut t_cur := f64(clamped_step)
	mut t_i := f64(s.t_0)
	if s.t_mult == 1 {
		t_cur = f64(clamped_step % s.t_0)
	} else {
		for t_cur >= t_i {
			t_cur -= t_i
			t_i *= f64(s.t_mult)
		}
	}
	return s.eta_min + (base_lr - s.eta_min) * (1.0 + math.cos(math.pi * t_cur / t_i)) / 2.0
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

// OneCycleLRConfig defines the learning-rate portion of the one-cycle policy.
// Momentum cycling and optimizer parameter groups are not part of this scheduler.
pub enum OneCycleAnnealStrategy {
	cosine
	linear
}

@[params]
pub struct OneCycleLRConfig {
pub:
	max_lr           f64
	total_steps      int
	pct_start        f64 = 0.3
	div_factor       f64 = 25.0
	final_div_factor f64 = 10000.0
	three_phase      bool
	anneal_strategy  OneCycleAnnealStrategy = .cosine
}

// OneCycleLR raises the rate from max_lr/div_factor to max_lr, then anneals it
// to max_lr/(div_factor*final_div_factor). Steps are minibatch steps, as in
// PyTorch's OneCycleLR. The optional three-phase policy returns to the initial
// rate before the final annealing phase.
pub struct OneCycleLR[T] {
	max_lr           f64
	total_steps      int
	pct_start        f64
	div_factor       f64
	final_div_factor f64
	three_phase      bool
	anneal_strategy  OneCycleAnnealStrategy
}

// one_cycle_lr creates a two-phase OneCycleLR by default. `anneal_strategy`
// selects cosine or linear interpolation. Invalid cycle parameters return an error.
pub fn one_cycle_lr[T](config OneCycleLRConfig) !&OneCycleLR[T] {
	if config.max_lr < 0.0 {
		return error('max_lr must be non-negative')
	}
	if config.total_steps <= 0 {
		return error('total_steps must be positive')
	}
	if config.pct_start <= 0.0 || config.pct_start >= 1.0 {
		return error('pct_start must be greater than 0 and less than 1')
	}
	if config.div_factor <= 0.0 || config.final_div_factor <= 0.0 {
		return error('div_factor and final_div_factor must be positive')
	}
	return &OneCycleLR[T]{
		max_lr:           config.max_lr
		total_steps:      config.total_steps
		pct_start:        config.pct_start
		div_factor:       config.div_factor
		final_div_factor: config.final_div_factor
		three_phase:      config.three_phase
		anneal_strategy:  config.anneal_strategy
	}
}

// next_lr returns the rate for a zero-based minibatch step. Steps outside the
// configured cycle are clamped to its first or final rate.
pub fn (s &OneCycleLR[T]) next_lr(_ f64, step int) f64 {
	initial_lr := s.max_lr / s.div_factor
	final_lr := initial_lr / s.final_div_factor
	if s.total_steps <= 1 {
		return final_lr
	}
	step_num := math.min(math.max(step, 0), s.total_steps - 1)
	first_end := s.pct_start * f64(s.total_steps) - 1.0
	if s.three_phase {
		second_end := 2.0 * s.pct_start * f64(s.total_steps) - 2.0
		if f64(step_num) <= first_end {
			return s.anneal(initial_lr, s.max_lr, 0.0, first_end, f64(step_num))
		}
		if f64(step_num) <= second_end {
			return s.anneal(s.max_lr, initial_lr, first_end, second_end, f64(step_num))
		}
		return s.anneal(initial_lr, final_lr, second_end, f64(s.total_steps - 1),
			f64(step_num))
	}
	if f64(step_num) <= first_end {
		return s.anneal(initial_lr, s.max_lr, 0.0, first_end, f64(step_num))
	}
	return s.anneal(s.max_lr, final_lr, first_end, f64(s.total_steps - 1), f64(step_num))
}

fn (s &OneCycleLR[T]) anneal(start f64, end f64, start_step f64, end_step f64, step f64) f64 {
	if end_step <= start_step {
		return end
	}
	pct := math.min(math.max((step - start_step) / (end_step - start_step), 0.0), 1.0)
	if s.anneal_strategy == .linear {
		return start + (end - start) * pct
	}
	return end + (start - end) * (1.0 + math.cos(math.pi * pct)) / 2.0
}

// CyclicLRMode selects the built-in cycle-amplitude policy.
pub enum CyclicLRMode {
	triangular
	triangular2
	exp_range
}

// CyclicLRConfig defines a minibatch-based triangular learning-rate schedule.
@[params]
pub struct CyclicLRConfig {
pub:
	base_lr        f64
	max_lr         f64
	step_size_up   int = 2000
	step_size_down int
	mode           CyclicLRMode = .triangular
	gamma          f64          = 1.0
	cycle_momentum bool         = true
	base_momentum  f64          = 0.8
	max_momentum   f64          = 0.9
}

// CyclicLR cycles the learning rate between base_lr and max_lr. Momentum
// can be queried as an inverse cycle with next_momentum. The scheduler does
// not update optimizer state or support parameter groups and custom scales.
pub struct CyclicLR[T] {
	base_lr        f64
	max_lr         f64
	step_size_up   int
	step_size_down int
	mode           CyclicLRMode
	gamma          f64
	cycle_momentum bool
	base_momentum  f64
	max_momentum   f64
}

// cyclic_lr creates a CyclicLR scheduler. A zero `step_size_down` uses the
// same duration as `step_size_up`; all other durations must be positive.
pub fn cyclic_lr[T](config CyclicLRConfig) !&CyclicLR[T] {
	if config.base_lr < 0.0 || config.max_lr < config.base_lr {
		return error('learning rates must satisfy 0 <= base_lr <= max_lr')
	}
	if config.step_size_up <= 0 || config.step_size_down < 0 {
		return error('step_size_up must be positive and step_size_down cannot be negative')
	}
	if config.gamma <= 0.0 {
		return error('gamma must be positive')
	}
	if config.base_momentum < 0.0 || config.max_momentum < config.base_momentum {
		return error('momentum values must satisfy 0 <= base_momentum <= max_momentum')
	}
	return &CyclicLR[T]{
		base_lr:        config.base_lr
		max_lr:         config.max_lr
		step_size_up:   config.step_size_up
		step_size_down: if config.step_size_down == 0 {
			config.step_size_up
		} else {
			config.step_size_down
		}
		mode:           config.mode
		gamma:          config.gamma
		cycle_momentum: config.cycle_momentum
		base_momentum:  config.base_momentum
		max_momentum:   config.max_momentum
	}
}

// next_momentum returns the inverse momentum curve for a zero-based minibatch
// step. It returns none when momentum cycling is disabled. Apply the returned
// value to an optimizer explicitly; this scheduler does not mutate optimizers.
pub fn (s &CyclicLR[T]) next_momentum(step int) ?f64 {
	if !s.cycle_momentum {
		return none
	}
	clamped_step := if step < 0 { 0 } else { step }
	period := s.step_size_up + s.step_size_down
	cycle := clamped_step / period
	step_in_cycle := clamped_step % period
	pct := if step_in_cycle <= s.step_size_up {
		f64(step_in_cycle) / f64(s.step_size_up)
	} else {
		1.0 - f64(step_in_cycle - s.step_size_up) / f64(s.step_size_down)
	}
	scale := match s.mode {
		.triangular { 1.0 }
		.triangular2 { math.pow(0.5, f64(cycle)) }
		.exp_range { math.pow(s.gamma, f64(clamped_step)) }
	}
	return s.max_momentum - (s.max_momentum - s.base_momentum) * pct * scale
}

// next_lr returns the rate for a zero-based minibatch step. The input rate is
// ignored because the schedule is defined by its base and maximum rates.
pub fn (s &CyclicLR[T]) next_lr(_ f64, step int) f64 {
	clamped_step := if step < 0 { 0 } else { step }
	period := s.step_size_up + s.step_size_down
	cycle := clamped_step / period
	step_in_cycle := clamped_step % period
	pct := if step_in_cycle <= s.step_size_up {
		f64(step_in_cycle) / f64(s.step_size_up)
	} else {
		1.0 - f64(step_in_cycle - s.step_size_up) / f64(s.step_size_down)
	}
	triangle := pct
	scale := match s.mode {
		.triangular { 1.0 }
		.triangular2 { math.pow(0.5, f64(cycle)) }
		.exp_range { math.pow(s.gamma, f64(clamped_step)) }
	}
	return s.base_lr + (s.max_lr - s.base_lr) * triangle * scale
}

// ReduceLROnPlateau reduces LR when a metric has stopped improving.
pub enum ReduceLROnPlateauMode {
	min
	max
}

pub enum ReduceLROnPlateauThresholdMode {
	rel
	abs
}

pub struct ReduceLROnPlateau[T] {
	factor         f64
	patience       int
	threshold      f64
	epsilon        f64
	cooldown       int
	mode           ReduceLROnPlateauMode
	threshold_mode ReduceLROnPlateauThresholdMode
	min_lr         f64
mut:
	cooldown_remaining      int
	step_cooldown_remaining int
	step_bad_epochs         int
	step_best               f64
pub mut:
	wait       int
	current_lr f64
}

// ReduceLROnPlateauConfig defines a public data structure for this module.

// ReduceLROnPlateauConfig defines a public data structure for this module.
@[params]
pub struct ReduceLROnPlateauConfig {
pub:
	factor         f64 = 0.1
	patience       int = 10
	threshold      f64 = 1e-4
	threshold_mode ReduceLROnPlateauThresholdMode = .rel
	mode           ReduceLROnPlateauMode          = .min
	epsilon        f64 = 1e-8
	cooldown       int
	min_lr         f64
}

// reduce_lr_on_plateau creates a ReduceLROnPlateau scheduler.
// Use `step` with the monitored metric for PyTorch-style behavior. `next_lr`
// remains available for callers that already provide a per-step metric delta.
pub fn reduce_lr_on_plateau[T](config ReduceLROnPlateauConfig) &ReduceLROnPlateau[T] {
	return try_reduce_lr_on_plateau[T](config) or { panic(err) }
}

// try_reduce_lr_on_plateau validates the configuration and creates a scheduler.
pub fn try_reduce_lr_on_plateau[T](config ReduceLROnPlateauConfig) !&ReduceLROnPlateau[T] {
	if config.factor <= 0.0 || config.factor >= 1.0 {
		return error('factor must be greater than 0 and less than 1')
	}
	if config.patience < 0 || config.cooldown < 0 || config.threshold < 0.0
		|| config.epsilon < 0.0 || config.min_lr < 0.0 {
		return error('patience, cooldown, threshold, epsilon, and min_lr cannot be negative')
	}
	return &ReduceLROnPlateau[T]{
		factor:                  config.factor
		patience:                config.patience
		threshold:               config.threshold
		epsilon:                 config.epsilon
		cooldown:                config.cooldown
		mode:                    config.mode
		threshold_mode:          config.threshold_mode
		min_lr:                  config.min_lr
		step_best:               if config.mode == .min { math.inf(1) } else { math.inf(-1) }
		wait:                    0
		step_bad_epochs:         0
		cooldown_remaining:      0
		step_cooldown_remaining: 0
	}
}

// step consumes a monitored metric and returns the PyTorch-style updated rate.
// A metric must beat the absolute or relative threshold to reset patience.
pub fn (mut s ReduceLROnPlateau[T]) step(current_lr f64, metric f64) f64 {
	s.current_lr = current_lr
	if s.metric_is_better(metric) {
		s.step_best = metric
		s.step_bad_epochs = 0
	} else {
		s.step_bad_epochs++
	}
	if s.step_cooldown_remaining > 0 {
		s.step_cooldown_remaining--
		s.step_bad_epochs = 0
	}
	if s.step_bad_epochs <= s.patience {
		return current_lr
	}
	s.step_bad_epochs = 0
	new_lr := math.max(current_lr * s.factor, s.min_lr)
	if current_lr - new_lr > s.epsilon {
		s.step_cooldown_remaining = s.cooldown
		s.current_lr = new_lr
		return new_lr
	}
	return current_lr
}

fn (s &ReduceLROnPlateau[T]) metric_is_better(metric f64) bool {
	if s.mode == .min {
		threshold := if s.threshold_mode == .rel {
			s.step_best * (1.0 - s.threshold)
		} else {
			s.step_best - s.threshold
		}
		return metric < threshold
	}
	threshold := if s.threshold_mode == .rel {
		s.step_best * (1.0 + s.threshold)
	} else {
		s.step_best + s.threshold
	}
	return metric > threshold
}

// next_lr preserves the original delta-based API. A negative delta beyond
// `-threshold` means improvement; other finite deltas count toward patience.
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
