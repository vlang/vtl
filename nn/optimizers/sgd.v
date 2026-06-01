module optimizers

import vtl.autograd
import vtl.nn.types
import vtl.nn.internal
import vtl

// SgdOptimizer implements vanilla Stochastic Gradient Descent with optional momentum.
pub struct SgdOptimizer[T] {
	learning_rate f64
pub mut:
	params []&autograd.Variable[T]
}

// SgdOptimizerConfig configures SgdOptimizer.
//
// Fields:
//   - `learning_rate` — step size α (default: 0.001)

// SgdOptimizerConfig defines a public data structure for this module.

// SgdOptimizerConfig defines a public data structure for this module.
@[params]
pub struct SgdOptimizerConfig {
pub:
	learning_rate f64 = 0.001
}

// sgd creates a new SgdOptimizer.
pub fn sgd[T](config SgdOptimizerConfig) &SgdOptimizer[T] {
	return &SgdOptimizer[T]{
		learning_rate: config.learning_rate
	}
}

// build_params registers all trainable variables from `layers`. Call once before training.
pub fn (mut o SgdOptimizer[T]) build_params(layers []types.Layer[T]) {
	for layer in layers {
		for v in layer.variables() {
			o.params << v
		}
	}
}

// update performs one SGD parameter update and zeros all gradients.
pub fn (mut o SgdOptimizer[T]) update() ! {
	for mut v in o.params {
		if v.requires_grad {
			internal.sgd_optimize[T](mut v.value, v.grad, o.learning_rate)!
			v.grad = vtl.zeros_like[T](v.value)
		}
	}
}
