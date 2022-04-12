module optimizers

import vtl.autograd
import vtl.nn.types
import vtl

pub struct AdamOptimizer<T> {
	learning_rate f64
	epsilon       f64
pub mut:
	beta1          f64
	beta2          f64
	beta1_t        f64
	beta2_t        f64
	params         []&autograd.Variable<T>
	first_moments  []&vtl.Tensor<T>
	second_moments []&vtl.Tensor<T>
}

[params]
pub struct AdamOptimizerConfig {
	learning_rate f64 = 0.001
	beta1         f64 = 0.9
	beta2         f64 = 0.999
	epsilon       f64 = 1e-8
}

pub fn new_adam_optimizer<T>(config AdamOptimizerConfig) &AdamOptimizer<T> {
	return &AdamOptimizer<T>{
		learning_rate: config.learning_rate
		beta1: config.beta1
		beta2: config.beta2
		epsilon: config.epsilon
	}
}

pub fn (mut o AdamOptimizer<T>) build_params(layes []types.Layer) {
	for layer in layers {
		for v in layer.variables() {
			o.params << v
			o.first_moments << vtl.zeros_like<T>(v.grad)
			o.second_moments << vtl.zeros_like<T>(v.grad)
		}
	}
}

pub fn (mut o AdamOptimizer<T>) update() {
	// @todo: @ulises-jeremias to implement this
}
