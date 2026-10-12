module optimizers

import vtl.autograd
import vtl.nn.types
import vtl.nn.internal
import vtl

// SgdOptimizer implements Stochastic Gradient Descent with optional momentum and weight decay.
pub struct SgdOptimizer[T] {
	learning_rate f64
	momentum      f64
	dampening     f64
	weight_decay  f64
	nesterov      bool
pub mut:
	params []&autograd.Variable[T]
mut:
	momentum_buffers     []&vtl.Tensor[T]
	momentum_initialized []bool
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

// update performs one SGD parameter update and zeros all gradients.
pub fn (mut o SgdOptimizer[T]) update() ! {
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
