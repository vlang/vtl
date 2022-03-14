module optimizers

import vtl.autograd
import vtl.nn
import vtl

pub struct SgdOptimizer<T> {
	learning_rate f64
pub mut:
	params []&autograd.Variable<T>
}

[params]
pub struct SgdOptimizerConfig {
	learning_rate f64 = 0.001
}

pub fn new_adam_optimizer<T>(config SgdOptimizerConfig) &SgdOptimizer<T> {
	return &SgdOptimizer<T>{
		learning_rate: config.learning_rate
	}
}

pub fn (mut o SgdOptimizer<T>) build_params(layes []nn.Layer) {
	for layer in layers {
		for v in layer.variables() {
			o.params << v
		}
	}
}

pub fn (mut o SgdOptimizer<T>) update() {
	// @todo: @ulises-jeremias to implement this
}
