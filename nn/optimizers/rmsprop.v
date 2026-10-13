module optimizers

import math
import vtl.autograd
import vtl.nn.types
import vtl

// RMSPropOptimizer implements the RMSProp optimiser.
//
// Maintains a running average of the squared gradient per parameter and
// normalises the update by it, allowing different effective learning rates
// per parameter.
//
// Update rule:
//   sq_avg = α·sq_avg + (1-α)·g²
//   θ = θ - lr · (g / (√sq_avg + ε) + wd·θ)
//
// Reference: Hinton, "Neural Networks for Machine Learning", Lecture 6e.
pub struct RMSPropOptimizer[T] {
	learning_rate f64
	epsilon       f64
pub mut:
	alpha        f64 // smoothing constant
	weight_decay f64
	params       []&autograd.Variable[T]
	sq_avg       []&vtl.Tensor[T]
	groups       []&RMSPropParamGroup[T]
}

// RMSPropOptimizerConfig configures RMSPropOptimizer.
//
// Fields:
//   - `learning_rate` — step size (default: 0.001)
//   - `alpha`         — smoothing constant for squared-gradient moving average (default: 0.99)
//   - `epsilon`       — numerical stability constant (default: 1e-8)
//   - `weight_decay`  — L2 regularisation coefficient (default: 0.0)

// RMSPropOptimizerConfig defines a public data structure for this module.

// RMSPropOptimizerConfig defines a public data structure for this module.
@[params]
pub struct RMSPropOptimizerConfig {
pub:
	learning_rate f64 = 0.001
	alpha         f64 = 0.99
	epsilon       f64 = 1e-8
	weight_decay  f64 = 0.0
}

// rmsprop creates a new RMSPropOptimizer.
pub fn rmsprop[T](config RMSPropOptimizerConfig) &RMSPropOptimizer[T] {
	return &RMSPropOptimizer[T]{
		learning_rate: config.learning_rate
		alpha:         config.alpha
		epsilon:       config.epsilon
		weight_decay:  config.weight_decay
	}
}

// checkpoint_config returns the scalar RMSProp state needed to resume training.
pub fn (o &RMSPropOptimizer[T]) checkpoint_config() map[string]f64 {
	return {
		'learning_rate': o.learning_rate
		'alpha':         o.alpha
		'epsilon':       o.epsilon
		'weight_decay':  o.weight_decay
	}
}

// restore_checkpoint_state restores RMSProp configuration and squared-gradient buffers.
pub fn (mut o RMSPropOptimizer[T]) restore_checkpoint_state(config map[string]f64,
	sq_avg []&vtl.Tensor[T]) ! {
	o.validate_checkpoint_state(config, sq_avg)!
	o.learning_rate = config['learning_rate']
	o.alpha = config['alpha']
	o.epsilon = config['epsilon']
	o.weight_decay = config['weight_decay']
	o.sq_avg = sq_avg.clone()
}

// validate_checkpoint_state checks RMSProp scalars and buffers without mutation.
pub fn (o &RMSPropOptimizer[T]) validate_checkpoint_state(config map[string]f64,
	sq_avg []&vtl.Tensor[T]) ! {
	for key in ['learning_rate', 'alpha', 'epsilon', 'weight_decay'] {
		if key !in config || !math.is_finite(config[key]) {
			return error('RMSPropOptimizer.restore_checkpoint_state: missing or non-finite ${key}')
		}
	}
	lr := config['learning_rate']
	alpha := config['alpha']
	epsilon := config['epsilon']
	weight_decay := config['weight_decay']
	if lr < 0 || alpha < 0 || alpha >= 1 || epsilon <= 0 || weight_decay < 0 {
		return error('RMSPropOptimizer.restore_checkpoint_state: scalar values are outside valid ranges')
	}
	if o.params.len != sq_avg.len {
		return error('RMSPropOptimizer.restore_checkpoint_state: parameter and buffer counts do not match')
	}
	for i, parameter in o.params {
		if sq_avg[i].shape != parameter.value.shape {
			return error('RMSPropOptimizer.restore_checkpoint_state: buffer shape mismatch at parameter ${i}')
		}
	}
}

// RMSPropParamGroup is a set of parameters sharing one RMSProp
// configuration, mirroring PyTorch parameter groups, with its own squared
// gradient averages.
pub struct RMSPropParamGroup[T] {
pub mut:
	params        []&autograd.Variable[T]
	learning_rate f64
	alpha         f64
	epsilon       f64
	weight_decay  f64
mut:
	sq_avg []&vtl.Tensor[T]
}

// add_param_group registers parameters under their own RMSProp configuration
// and returns the group index. Register each variable either flat (via
// build_params) or in exactly one group.
pub fn (mut o RMSPropOptimizer[T]) add_param_group(params []&autograd.Variable[T], config RMSPropOptimizerConfig) !int {
	if config.learning_rate < 0 || config.alpha < 0 || config.alpha >= 1 || config.epsilon <= 0
		|| config.weight_decay < 0 {
		return error('RMSProp group learning rate must be non-negative, alpha must lie in [0, 1), epsilon must be positive, and weight decay must be non-negative')
	}
	mut group := &RMSPropParamGroup[T]{
		learning_rate: config.learning_rate
		alpha:         config.alpha
		epsilon:       config.epsilon
		weight_decay:  config.weight_decay
	}
	for v in params {
		group.params << v
		group.sq_avg << vtl.zeros_like[T](v.grad)
	}
	o.groups << group
	return o.groups.len - 1
}

// group_count returns how many parameter groups are registered.
pub fn (o &RMSPropOptimizer[T]) group_count() int {
	return o.groups.len
}

// group_learning_rate returns the current learning rate of one group.
pub fn (o &RMSPropOptimizer[T]) group_learning_rate(group_index int) !f64 {
	if group_index < 0 || group_index >= o.groups.len {
		return error('RMSProp group index ${group_index} out of range for ${o.groups.len} groups')
	}
	return o.groups[group_index].learning_rate
}

// set_group_learning_rate applies a new learning rate to one group only;
// this is the integration point for learning-rate schedulers.
pub fn (mut o RMSPropOptimizer[T]) set_group_learning_rate(group_index int, learning_rate f64) ! {
	if learning_rate < 0 {
		return error('RMSProp group learning rate must be non-negative')
	}
	if group_index < 0 || group_index >= o.groups.len {
		return error('RMSProp group index ${group_index} out of range for ${o.groups.len} groups')
	}
	o.groups[group_index].learning_rate = learning_rate
}

fn (mut g RMSPropParamGroup[T]) update_group() ! {
	for i, mut v in g.params {
		if v.requires_grad {
			g.sq_avg[i].napply([v.grad], fn [g] [T](vals []T, idx []int) T {
				grad := f64(vals[1])
				return vtl.cast[T](g.alpha * f64(vals[0]) + (1.0 - g.alpha) * grad * grad)
			}) or { return err }
			sq_avg_i := g.sq_avg[i]
			v.value.napply([v.grad, sq_avg_i], fn [g] [T](vals []T, idx []int) T {
				theta := f64(vals[0])
				grad := f64(vals[1])
				vv := f64(vals[2])
				return vtl.cast[T](theta - g.learning_rate * (grad / (math.sqrt(vv) + g.epsilon) +
					g.weight_decay * theta))
			}) or { return err }
			v.grad = vtl.zeros_like[T](v.value)
		}
	}
}

// build_params registers all trainable variables from `layers`. Call once before training.
pub fn (mut o RMSPropOptimizer[T]) build_params(layers []types.Layer[T]) {
	for layer in layers {
		for v in layer.variables() {
			o.params << v
			o.sq_avg << vtl.zeros_like[T](v.grad)
		}
	}
}

// update performs one RMSProp parameter update and zeros all gradients.
pub fn (mut o RMSPropOptimizer[T]) update() ! {
	for mut group in o.groups {
		group.update_group()!
	}
	for i, mut v in o.params {
		if v.requires_grad {
			// sq_avg = alpha * sq_avg + (1-alpha) * grad^2
			o.sq_avg[i].napply([v.grad], fn [o] [T](vals []T, idx []int) T {
				g := f64(vals[1])
				return vtl.cast[T](o.alpha * f64(vals[0]) + (1.0 - o.alpha) * g * g)
			}) or { return err }

			// theta = theta - lr * (grad / (sqrt(sq_avg) + eps) + wd * theta)
			sq_avg_i := o.sq_avg[i]
			v.value.napply([v.grad, sq_avg_i], fn [o] [T](vals []T, idx []int) T {
				theta := f64(vals[0])
				grad := f64(vals[1])
				vv := f64(vals[2])
				return vtl.cast[T](theta - o.learning_rate * (grad / (math.sqrt(vv) + o.epsilon) +
					o.weight_decay * theta))
			}) or { return err }

			v.grad = vtl.zeros_like[T](v.value)
		}
	}
}
