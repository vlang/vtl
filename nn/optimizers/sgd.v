module optimizers

import vtl.autograd
import vtl.nn.types
import vtl.nn.internal
import vtl

pub struct SgdOptimizer[T] {
	learning_rate f64
pub mut:
	params []&autograd.Variable[T]
}

@[params]
pub struct SgdOptimizerConfig {
pub:
	learning_rate f64 = 0.001
}

pub fn sgd[T](config SgdOptimizerConfig) &SgdOptimizer[T] {
	return &SgdOptimizer[T]{
		learning_rate: config.learning_rate
	}
}

pub fn (mut o SgdOptimizer[T]) build_params(layers []types.Layer[T]) {
	for layer in layers {
		for v in layer.variables() {
			o.params << v
		}
	}
}

pub fn (mut o SgdOptimizer[T]) update() ! {
	for mut v in o.params {
		if v.requires_grad {
			internal.sgd_optimize[T](mut v.value, v.grad, o.learning_rate)!
			v.grad = vtl.zeros_like[T](v.value)
		}
	}
}
