module optimizers

import vtl.autograd
import vtl

pub struct AdamOptimizer<T> {
	params        []&autograd.Variable<T>
	learning_rate f64
	beta1         f64
	beta2         f64
	epsilon       f64
	beta1_t       f64
	beta2_t       f64
	first_moment  []&vtl.Tensor<T>
	second_moment []&vtl.Tensor<T>
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
