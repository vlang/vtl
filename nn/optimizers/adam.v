module optimizers

import math
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

pub fn adam_optimizer<T>(config AdamOptimizerConfig) &AdamOptimizer<T> {
	return &AdamOptimizer<T>{
		learning_rate: config.learning_rate
		beta1: config.beta1
		beta2: config.beta2
		epsilon: config.epsilon
	}
}

pub fn (mut o AdamOptimizer<T>) build_params(layers []types.Layer) {
	// @todo: @ulises-jeremias to uncomment this
	for layer in layers {
		// for v in layer.variables() {
		// 	o.params << v
		// 	o.first_moments << vtl.zeros_like<T>(v.grad)
		// 	o.second_moments << vtl.zeros_like<T>(v.grad)
		// }
	}
}

pub fn (mut o AdamOptimizer<T>) update() ? {
	lr_t := o.learning_rate * math.sqrt(1.0 - o.beta2_t) / (1.0 - o.beta1_t)

	o.beta1_t *= o.beta1
	o.beta2_t *= o.beta2

	for i, mut v in o.params {
		if v.requires_grad {
			mut fm_iters, _ := o.first_moments[i].iterators([v.grad])?
			for {
				vals, idx := fm_iters.next() or { break }
				val := o.beta1 * vals[0] + (1.0 - o.beta1) * vals[1]
				o.first_moments[i].set(idx, vtl.new_t<T>(val))
			}

			mut sm_iters, _ := o.second_moments[i].iterators([v.grad])?
			for {
				vals, idx := sm_iters.next() or { break }
				val := o.beta2 * vals[0] + (1.0 - o.beta2) * vals[1] * vals[1]
				o.second_moments[i].set(idx, vtl.new_t<T>(val))
			}

			mut val_iters, _ := v.value.iterators([o.first_moments[i], o.second_moments[i]])?
			for {
				vals, idx := val_iters.next() or { break }
				val := vals[0] - lr_t * vals[1] / (math.sqrt(vals[3]) + o.epsilon)
				v.value.set(idx, vtl.new_t<T>(val))
			}

			v.grad = vtl.zeros_like<T>(v.value)
		}
	}
}
