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
}

// RMSPropOptimizerConfig configures RMSPropOptimizer.
//
// Fields:
//   - `learning_rate` — step size (default: 0.001)
//   - `alpha`         — smoothing constant for squared-gradient moving average (default: 0.99)
//   - `epsilon`       — numerical stability constant (default: 1e-8)
//   - `weight_decay`  — L2 regularisation coefficient (default: 0.0)
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
