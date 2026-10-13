module optimizers

import vtl.autograd
import vtl.nn.types
import vtl.nn.internal
import vtl
import math

// SgdOptimizer implements Stochastic Gradient Descent with optional momentum and weight decay.
pub struct SgdOptimizer[T] {
	learning_rate f64
	momentum      f64
	dampening     f64
	weight_decay  f64
	nesterov      bool
pub mut:
	params               []&autograd.Variable[T]
	momentum_buffers     []&vtl.Tensor[T]
	momentum_initialized []bool
	groups               []&SgdParamGroup[T]
}

// SgdOptimizerConfig configures SgdOptimizer.
//
// Fields:
//   - `learning_rate` — step size α (default: 0.001)
//   - `momentum` — momentum factor (default: 0.0)
//   - `dampening` — dampens new gradients in the momentum buffer (default: 0.0)
//   - `weight_decay` — coupled L2 regularization coefficient (default: 0.0)
//   - `nesterov` — use the Nesterov update (default: false)

// SgdOptimizerConfig defines a public data structure for this module.
@[params]
pub struct SgdOptimizerConfig {
pub:
	learning_rate f64 = 0.001
	momentum      f64
	dampening     f64
	weight_decay  f64
	nesterov      bool
}

// SgdParamGroup is a set of parameters sharing one SGD configuration,
// mirroring PyTorch parameter groups. Groups are updated with their own
// learning rate, momentum, dampening, weight decay and Nesterov settings,
// independently of the optimizer-wide defaults used by the flat params list.
pub struct SgdParamGroup[T] {
pub mut:
	params        []&autograd.Variable[T]
	learning_rate f64
	momentum      f64
	dampening     f64
	weight_decay  f64
	nesterov      bool
mut:
	momentum_buffers     []&vtl.Tensor[T]
	momentum_initialized []bool
}

// sgd creates a new SgdOptimizer.
pub fn sgd[T](config SgdOptimizerConfig) &SgdOptimizer[T] {
	return &SgdOptimizer[T]{
		learning_rate: config.learning_rate
		momentum:      config.momentum
		dampening:     config.dampening
		weight_decay:  config.weight_decay
		nesterov:      config.nesterov
	}
}

// checkpoint_config returns the scalar configuration and per-parameter
// momentum initialization state needed to resume SGD exactly.
pub fn (o &SgdOptimizer[T]) checkpoint_config() map[string]f64 {
	mut config := {
		'learning_rate': o.learning_rate
		'momentum':      o.momentum
		'dampening':     o.dampening
		'weight_decay':  o.weight_decay
		'nesterov':      if o.nesterov { 1.0 } else { 0.0 }
	}
	if o.momentum > 0 {
		for i, initialized in o.momentum_initialized {
			config['momentum_initialized_${i}'] = if initialized { 1.0 } else { 0.0 }
		}
	}
	return config
}

// restore_checkpoint_state restores SGD configuration, momentum buffers, and
// their initialization state. Parameters must already be registered in order.
pub fn (mut o SgdOptimizer[T]) restore_checkpoint_state(config map[string]f64,
	momentum_buffers []&vtl.Tensor[T]) ! {
	o.validate_checkpoint_state(config, momentum_buffers)!
	o.learning_rate = config['learning_rate']
	o.momentum = config['momentum']
	o.dampening = config['dampening']
	o.weight_decay = config['weight_decay']
	o.nesterov = config['nesterov'] == 1.0
	o.momentum_buffers = momentum_buffers.clone()
	o.momentum_initialized = []bool{cap: o.params.len}
	if o.momentum > 0 {
		for i in 0 .. o.params.len {
			o.momentum_initialized << config['momentum_initialized_${i}'] == 1.0
		}
	}
}

// validate_checkpoint_state checks SGD configuration and momentum state
// without changing the optimizer.
pub fn (o &SgdOptimizer[T]) validate_checkpoint_state(config map[string]f64,
	momentum_buffers []&vtl.Tensor[T]) ! {
	for key in ['learning_rate', 'momentum', 'dampening', 'weight_decay', 'nesterov'] {
		if key !in config || !math.is_finite(config[key]) {
			return error('SgdOptimizer.restore_checkpoint_state: missing or non-finite ${key}')
		}
	}
	lr := config['learning_rate']
	momentum := config['momentum']
	dampening := config['dampening']
	weight_decay := config['weight_decay']
	nesterov_value := config['nesterov']
	if lr < 0 || momentum < 0 || dampening < 0 || weight_decay < 0
		|| nesterov_value !in [0.0, 1.0] {
		return error('SgdOptimizer.restore_checkpoint_state: scalar values are outside valid ranges')
	}
	if nesterov_value == 1.0 && (momentum <= 0 || dampening != 0) {
		return error('SgdOptimizer.restore_checkpoint_state: invalid Nesterov configuration')
	}
	if momentum == 0 {
		if momentum_buffers.len != 0 {
			return error('SgdOptimizer.restore_checkpoint_state: unexpected momentum buffers')
		}
		return
	}
	if momentum_buffers.len != o.params.len {
		return error('SgdOptimizer.restore_checkpoint_state: parameter and momentum counts do not match')
	}
	for i, parameter in o.params {
		initialized_key := 'momentum_initialized_${i}'
		if initialized_key !in config || config[initialized_key] !in [0.0, 1.0] {
			return error('SgdOptimizer.restore_checkpoint_state: invalid momentum state at parameter ${i}')
		}
		if momentum_buffers[i].shape != parameter.value.shape {
			return error('SgdOptimizer.restore_checkpoint_state: momentum shape mismatch at parameter ${i}')
		}
	}
}

// build_params registers all trainable variables from `layers`. Call once before training.
pub fn (mut o SgdOptimizer[T]) build_params(layers []types.Layer[T]) {
	for layer in layers {
		for v in layer.variables() {
			o.params << v
			if o.momentum > 0 {
				o.momentum_buffers << vtl.zeros_like[T](v.grad)
				o.momentum_initialized << false
			}
		}
	}
}

// add_param_group registers parameters under their own SGD configuration and
// returns the group index. Register each variable either flat (via params or
// build_params) or in exactly one group. Group momentum buffers are created
// here, matching build_params behavior for the flat list.
pub fn (mut o SgdOptimizer[T]) add_param_group(params []&autograd.Variable[T], config SgdOptimizerConfig) !int {
	if config.learning_rate < 0 || config.momentum < 0 || config.dampening < 0
		|| config.weight_decay < 0 {
		return error('SGD group learning rate, momentum, dampening, and weight decay must be non-negative')
	}
	if config.nesterov && (config.momentum <= 0 || config.dampening != 0) {
		return error('SGD group Nesterov momentum requires positive momentum and zero dampening')
	}
	mut group := &SgdParamGroup[T]{
		learning_rate: config.learning_rate
		momentum:      config.momentum
		dampening:     config.dampening
		weight_decay:  config.weight_decay
		nesterov:      config.nesterov
	}
	for v in params {
		group.params << v
		if config.momentum > 0 {
			group.momentum_buffers << vtl.zeros_like[T](v.grad)
			group.momentum_initialized << false
		}
	}
	o.groups << group
	return o.groups.len - 1
}

// group_count returns how many parameter groups are registered.
pub fn (o &SgdOptimizer[T]) group_count() int {
	return o.groups.len
}

// group_learning_rate returns the current learning rate of one group.
pub fn (o &SgdOptimizer[T]) group_learning_rate(group_index int) !f64 {
	if group_index < 0 || group_index >= o.groups.len {
		return error('SGD group index ${group_index} out of range for ${o.groups.len} groups')
	}
	return o.groups[group_index].learning_rate
}

// set_group_learning_rate applies a new learning rate to one group only.
// This is the integration point for learning-rate schedulers, which compute
// the next rate and hand it back for the caller to apply.
pub fn (mut o SgdOptimizer[T]) set_group_learning_rate(group_index int, learning_rate f64) ! {
	if learning_rate < 0 {
		return error('SGD group learning rate must be non-negative')
	}
	if group_index < 0 || group_index >= o.groups.len {
		return error('SGD group index ${group_index} out of range for ${o.groups.len} groups')
	}
	o.groups[group_index].learning_rate = learning_rate
}

fn (mut g SgdParamGroup[T]) update_group() ! {
	if g.momentum > 0 && (g.momentum_buffers.len != g.params.len
		|| g.momentum_initialized.len != g.params.len) {
		return error('SGD group momentum requires parameters registered with add_param_group')
	}
	for i, mut v in g.params {
		if v.requires_grad {
			if g.momentum > 0 {
				mut iters, _ := v.value.iterators([v.grad, g.momentum_buffers[i]])!
				for {
					vals, index := iters.next() or { break }
					theta := f64(vals[0])
					regularized_grad := f64(vals[1]) + g.weight_decay * theta
					mut buffer := f64(vals[2])
					if g.momentum_initialized[i] {
						buffer = g.momentum * buffer + (1.0 - g.dampening) * regularized_grad
					} else {
						buffer = regularized_grad
					}
					g.momentum_buffers[i].set(index, vtl.cast[T](buffer))
					direction := if g.nesterov {
						regularized_grad + g.momentum * buffer
					} else {
						buffer
					}
					v.value.set(index, vtl.cast[T](theta - g.learning_rate * direction))
				}
				g.momentum_initialized[i] = true
			} else if g.weight_decay > 0 {
				mut iters, _ := v.value.iterators([v.grad])!
				for {
					vals, index := iters.next() or { break }
					theta := f64(vals[0])
					gradient := f64(vals[1]) + g.weight_decay * theta
					v.value.set(index, vtl.cast[T](theta - g.learning_rate * gradient))
				}
			} else {
				internal.sgd_optimize[T](mut v.value, v.grad, g.learning_rate)!
			}
			v.grad = vtl.zeros_like[T](v.value)
		}
	}
}

// update performs one SGD parameter update and zeros all gradients.
pub fn (mut o SgdOptimizer[T]) update() ! {
	for mut group in o.groups {
		group.update_group()!
	}
	if o.learning_rate < 0 || o.momentum < 0 || o.dampening < 0 || o.weight_decay < 0 {
		return error('SGD learning rate, momentum, dampening, and weight decay must be non-negative')
	}
	if o.nesterov && (o.momentum <= 0 || o.dampening != 0) {
		return error('SGD Nesterov momentum requires positive momentum and zero dampening')
	}
	if o.momentum > 0 && (o.momentum_buffers.len != o.params.len
		|| o.momentum_initialized.len != o.params.len) {
		return error('SGD momentum requires parameters registered with build_params')
	}
	for i, mut v in o.params {
		if v.requires_grad {
			if o.momentum > 0 {
				mut iters, _ := v.value.iterators([v.grad, o.momentum_buffers[i]])!
				for {
					vals, index := iters.next() or { break }
					theta := f64(vals[0])
					regularized_grad := f64(vals[1]) + o.weight_decay * theta
					mut buffer := f64(vals[2])
					if o.momentum_initialized[i] {
						buffer = o.momentum * buffer + (1.0 - o.dampening) * regularized_grad
					} else {
						// PyTorch initializes the first buffer directly from the gradient.
						buffer = regularized_grad
					}
					o.momentum_buffers[i].set(index, vtl.cast[T](buffer))
					direction := if o.nesterov {
						regularized_grad + o.momentum * buffer
					} else {
						buffer
					}
					v.value.set(index, vtl.cast[T](theta - o.learning_rate * direction))
				}
				o.momentum_initialized[i] = true
			} else if o.weight_decay > 0 {
				mut iters, _ := v.value.iterators([v.grad])!
				for {
					vals, index := iters.next() or { break }
					theta := f64(vals[0])
					gradient := f64(vals[1]) + o.weight_decay * theta
					v.value.set(index, vtl.cast[T](theta - o.learning_rate * gradient))
				}
			} else {
				internal.sgd_optimize[T](mut v.value, v.grad, o.learning_rate)!
			}
			v.grad = vtl.zeros_like[T](v.value)
		}
	}
}
