module amp

import math
import vtl

// GradScalerConfig follows PyTorch's gradient-scaler defaults while staying
// backend-independent: callers apply the scale to a loss and pass gradients to
// unscale before an optimizer step.
@[params]
pub struct GradScalerConfig {
pub:
	init_scale      f64  = 65536.0
	growth_factor   f64  = 2.0
	backoff_factor  f64  = 0.5
	growth_interval int  = 2000
	enabled         bool = true
}

// GradScalerState is the data-only checkpoint form.
pub struct GradScalerState {
pub:
	scale      f64
	good_steps int
	enabled    bool
}

// GradScaler performs loss scaling, gradient unscaling, overflow detection,
// and scale updates.
pub struct GradScaler {
mut:
	scale           f64
	good_steps      int
	growth_factor   f64
	backoff_factor  f64
	growth_interval int
	enabled         bool
}

// new validates a scaler configuration.
pub fn new(config GradScalerConfig) !&GradScaler {
	if !config.enabled {
		return &GradScaler{
			scale:           1.0
			growth_factor:   config.growth_factor
			backoff_factor:  config.backoff_factor
			growth_interval: config.growth_interval
			enabled:         false
		}
	}
	if config.init_scale <= 0.0 || !math.is_finite(config.init_scale) {
		return error('amp: init_scale must be finite and positive')
	}
	if config.growth_factor <= 1.0 || !math.is_finite(config.growth_factor) {
		return error('amp: growth_factor must be finite and greater than 1')
	}
	if config.backoff_factor <= 0.0 || config.backoff_factor >= 1.0 || !math.is_finite(config.backoff_factor) {
		return error('amp: backoff_factor must be finite and between 0 and 1')
	}
	if config.growth_interval <= 0 {
		return error('amp: growth_interval must be positive')
	}
	return &GradScaler{
		scale:           config.init_scale
		growth_factor:   config.growth_factor
		backoff_factor:  config.backoff_factor
		growth_interval: config.growth_interval
		enabled:         true
	}
}

// scale_loss multiplies a loss by the current scale.
pub fn (s &GradScaler) scale_loss(loss f64) f64 {
	if !s.enabled {
		return loss
	}
	return loss * s.scale
}

// scale_factor reports the current scale.
pub fn (s &GradScaler) scale_factor() f64 {
	return s.scale
}

// unscale divides every gradient by the current scale in place. It returns
// false when any gradient is non-finite; those values are left untouched so
// the caller can skip the optimizer step and call update with overflow.
pub fn (mut s GradScaler) unscale[T](mut gradients []&vtl.Tensor[T]) !bool {
	if !s.enabled {
		return true
	}
	factor := 1.0 / s.scale
	mut overflow := false
	for tensor in gradients {
		for i in 0 .. tensor.size {
			value := tensor.get_nth[T](i)
			as_f64 := f64(vtl.cast[f64](value))
			if !math.is_finite(as_f64) {
				overflow = true
				continue
			}
			candidate := as_f64 * factor
			if !math.is_finite(candidate) {
				overflow = true
				continue
			}
			narrowed := vtl.cast[T](candidate)
			back := f64(vtl.cast[f64](narrowed))
			if !math.is_finite(back) {
				overflow = true
				continue
			}
			tensor.set_nth[T](i, narrowed)
		}
	}
	return !overflow
}

// update advances the scaler after a step. An overflow backs the scale off and
// resets the good-step counter; otherwise the scale grows after the configured
// number of consecutive good steps.
pub fn (mut s GradScaler) update(overflow bool) {
	if !s.enabled {
		return
	}
	if overflow {
		s.scale *= s.backoff_factor
		s.good_steps = 0
		return
	}
	s.good_steps++
	if s.good_steps >= s.growth_interval {
		s.scale *= s.growth_factor
		s.good_steps = 0
	}
}

// state returns the checkpointable scaler state.
pub fn (s &GradScaler) state() GradScalerState {
	return GradScalerState{
		scale:      s.scale
		good_steps: s.good_steps
		enabled:    s.enabled
	}
}

// from_state restores a scaler state created by state.
// The saved `state.enabled` flag wins: if the checkpoint was taken from a
// disabled scaler, the restored scaler stays disabled even when
// `config.enabled` is true.
pub fn from_state(state GradScalerState, config GradScalerConfig) !&GradScaler {
	if state.scale <= 0.0 || !math.is_finite(state.scale) {
		return error('amp: restored scale must be finite and positive')
	}
	if state.good_steps < 0 {
		return error('amp: restored good_steps must be non-negative')
	}
	mut scaler := new(config)!
	scaler.scale = state.scale
	scaler.good_steps = state.good_steps
	scaler.enabled = state.enabled
	return scaler
}
