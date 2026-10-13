module optimizers

import math
import vtl.autograd
import vtl.nn.types
import vtl

// AdaGradOptimizer implements the AdaGrad (Adaptive Gradient) algorithm.
// Accumulates squared gradients and adapts the learning rate per parameter.
pub struct AdaGradOptimizer[T] {
	learning_rate f64
	epsilon       f64
pub mut:
	weight_decay         f64
	params               []&autograd.Variable[T]
	accumulated_sq_grads []&vtl.Tensor[T]
	groups               []&AdaGradParamGroup[T]
}

// AdaGradOptimizerConfig defines a public data structure for this module.

// AdaGradOptimizerConfig defines a public data structure for this module.
@[params]
pub struct AdaGradOptimizerConfig {
pub:
	learning_rate f64 = 0.01
	epsilon       f64 = 1e-8
	weight_decay  f64 = 0.0
}

// adagrad creates a new AdaGradOptimizer.
pub fn adagrad[T](config AdaGradOptimizerConfig) &AdaGradOptimizer[T] {
	return &AdaGradOptimizer[T]{
		learning_rate: config.learning_rate
		epsilon:       config.epsilon
		weight_decay:  config.weight_decay
	}
}

// checkpoint_config returns the scalar AdaGrad state needed to resume training.
pub fn (o &AdaGradOptimizer[T]) checkpoint_config() map[string]f64 {
	return {
		'learning_rate': o.learning_rate
		'epsilon':       o.epsilon
		'weight_decay':  o.weight_decay
	}
}

// restore_checkpoint_state restores AdaGrad configuration and accumulated gradients.
pub fn (mut o AdaGradOptimizer[T]) restore_checkpoint_state(config map[string]f64,
	accumulated_sq_grads []&vtl.Tensor[T]) ! {
	o.validate_checkpoint_state(config, accumulated_sq_grads)!
	o.learning_rate = config['learning_rate']
	o.epsilon = config['epsilon']
	o.weight_decay = config['weight_decay']
	o.accumulated_sq_grads = accumulated_sq_grads.clone()
}

// validate_checkpoint_state checks AdaGrad scalars and accumulators without mutation.
pub fn (o &AdaGradOptimizer[T]) validate_checkpoint_state(config map[string]f64,
	accumulated_sq_grads []&vtl.Tensor[T]) ! {
	for key in ['learning_rate', 'epsilon', 'weight_decay'] {
		if key !in config || !math.is_finite(config[key]) {
			return error('AdaGradOptimizer.restore_checkpoint_state: missing or non-finite ${key}')
		}
	}
	lr := config['learning_rate']
	epsilon := config['epsilon']
	weight_decay := config['weight_decay']
	if lr < 0 || epsilon <= 0 || weight_decay < 0 {
		return error('AdaGradOptimizer.restore_checkpoint_state: scalar values are outside valid ranges')
	}
	if o.params.len != accumulated_sq_grads.len {
		return error('AdaGradOptimizer.restore_checkpoint_state: parameter and accumulator counts do not match')
	}
	for i, parameter in o.params {
		if accumulated_sq_grads[i].shape != parameter.value.shape {
			return error('AdaGradOptimizer.restore_checkpoint_state: accumulator shape mismatch at parameter ${i}')
		}
	}
}

// AdaGradParamGroup is a set of parameters sharing one AdaGrad
// configuration, mirroring PyTorch parameter groups, with its own
// accumulated squared-gradient buffers.
pub struct AdaGradParamGroup[T] {
pub mut:
	params               []&autograd.Variable[T]
	learning_rate        f64
	epsilon              f64
	weight_decay         f64
	accumulated_sq_grads []&vtl.Tensor[T]
}

// add_param_group registers parameters under their own AdaGrad configuration
// and returns the group index. Register each variable either flat (via
// build_params) or in exactly one group.
pub fn (mut o AdaGradOptimizer[T]) add_param_group(params []&autograd.Variable[T], config AdaGradOptimizerConfig) !int {
	if config.learning_rate < 0 || config.epsilon <= 0 || config.weight_decay < 0 {
		return error('AdaGrad group learning rate and weight decay must be non-negative and epsilon must be positive')
	}
	mut group := &AdaGradParamGroup[T]{
		learning_rate: config.learning_rate
		epsilon:       config.epsilon
		weight_decay:  config.weight_decay
	}
	for v in params {
		group.params << v
		group.accumulated_sq_grads << vtl.zeros_like[T](v.grad)
	}
	o.groups << group
	return o.groups.len - 1
}

// group_count returns how many parameter groups are registered.
pub fn (o &AdaGradOptimizer[T]) group_count() int {
	return o.groups.len
}

// group_learning_rate returns the current learning rate of one group.
pub fn (o &AdaGradOptimizer[T]) group_learning_rate(group_index int) !f64 {
	if group_index < 0 || group_index >= o.groups.len {
		return error('AdaGrad group index ${group_index} out of range for ${o.groups.len} groups')
	}
	return o.groups[group_index].learning_rate
}

// set_group_learning_rate applies a new learning rate to one group only;
// this is the integration point for learning-rate schedulers.
pub fn (mut o AdaGradOptimizer[T]) set_group_learning_rate(group_index int, learning_rate f64) ! {
	if learning_rate < 0 {
		return error('AdaGrad group learning rate must be non-negative')
	}
	if group_index < 0 || group_index >= o.groups.len {
		return error('AdaGrad group index ${group_index} out of range for ${o.groups.len} groups')
	}
	o.groups[group_index].learning_rate = learning_rate
}

fn (mut g AdaGradParamGroup[T]) update_group() ! {
	for i, mut v in g.params {
		if v.requires_grad {
			// napply passes the accumulator itself as vals[0] and the
			// gradient as vals[1]: G = G + grad^2.
			g.accumulated_sq_grads[i].napply([v.grad], fn [T](vals []T, idx []int) T {
				grad := f64(vals[1])
				return vtl.cast[T](f64(vals[0]) + grad * grad)
			}) or { return err }

			v.value.napply([g.accumulated_sq_grads[i], v.grad], fn [g] [T](vals []T, idx []int) T {
				theta := f64(vals[0])
				acc := f64(vals[1])
				grad := f64(vals[2])
				return vtl.cast[T](theta - g.learning_rate * (grad / (math.sqrt(acc) + g.epsilon) +
					g.weight_decay * theta))
			}) or { return err }

			v.grad = vtl.zeros_like[T](v.value)
		}
	}
}

// build_params registers all trainable variables from `layers`. Call once before training.
pub fn (mut o AdaGradOptimizer[T]) build_params(layers []types.Layer[T]) {
	for layer in layers {
		for v in layer.variables() {
			o.params << v
			o.accumulated_sq_grads << vtl.zeros_like[T](v.grad)
		}
	}
}

// update performs one AdaGrad parameter update and zeros all gradients.
pub fn (mut o AdaGradOptimizer[T]) update() ! {
	for mut group in o.groups {
		group.update_group()!
	}
	for i, mut v in o.params {
		if v.requires_grad {
			// Accumulate squared gradients: G += grad^2
			// napply passes the accumulator itself as vals[0] and the
			// gradient as vals[1]: G = G + grad^2.
			o.accumulated_sq_grads[i].napply([v.grad], fn [T](vals []T, idx []int) T {
				grad := f64(vals[1])
				return vtl.cast[T](f64(vals[0]) + grad * grad)
			}) or { return err }

			// theta = theta - lr * (grad / (sqrt(G) + eps) + wd * theta)
			v.value.napply([o.accumulated_sq_grads[i], v.grad], fn [o] [T](vals []T, idx []int) T {
				theta := f64(vals[0])
				g := f64(vals[1])
				grad := f64(vals[2])
				return vtl.cast[T](theta - o.learning_rate * (grad / (math.sqrt(g) + o.epsilon) +
					o.weight_decay * theta))
			}) or { return err }

			v.grad = vtl.zeros_like[T](v.value)
		}
	}
}
