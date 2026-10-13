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
	for i, mut v in o.params {
		if v.requires_grad {
			// Accumulate squared gradients: G += grad^2
			o.accumulated_sq_grads[i].napply([v.grad], fn [T](vals []T, idx []int) T {
				grad := f64(vals[0])
				return vtl.cast[T](f64(vals[1]) + grad * grad)
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
