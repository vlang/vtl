module optimizers

import math
import vtl.autograd
import vtl.nn.types
import vtl

pub struct AdamOptimizer[T] {
	learning_rate f64
	epsilon       f64
pub mut:
	beta1          f64
	beta2          f64
	beta1_t        f64
	beta2_t        f64
	params         []&autograd.Variable[T]
	first_moments  []&vtl.Tensor[T]
	second_moments []&vtl.Tensor[T]
}

@[params]
pub struct AdamOptimizerConfig {
	learning_rate f64 = 0.001
	beta1         f64 = 0.9
	beta2         f64 = 0.999
	epsilon       f64 = 1e-8
}

pub fn adam_optimizer[T](config AdamOptimizerConfig) &AdamOptimizer[T] {
	return &AdamOptimizer[T]{
		learning_rate: config.learning_rate
		beta1:         config.beta1
		beta2:         config.beta2
		epsilon:       config.epsilon
		beta1_t:       config.beta1
		beta2_t:       config.beta2
	}
}

pub fn (mut o AdamOptimizer[T]) build_params(layers []types.Layer[T]) {
	for layer in layers {
		for v in layer.variables() {
			o.params << v
			o.first_moments << vtl.zeros_like[T](v.grad)
			o.second_moments << vtl.zeros_like[T](v.grad)
		}
	}
}

pub fn (mut o AdamOptimizer[T]) update() ! {
	lr_t := o.learning_rate * math.sqrt(1.0 - o.beta2_t) / (1.0 - o.beta1_t)

	o.beta1_t *= o.beta1
	o.beta2_t *= o.beta2

	for i, mut v in o.params {
		if v.requires_grad {
			// m = beta1 * m + (1 - beta1) * grad
			o.first_moments[i].napply([v.grad], fn [o] [T](vals []T, idx []int) T {
				return vtl.cast[T](o.beta1 * f64(vals[0]) + (1.0 - o.beta1) * f64(vals[1]))
			}) or { return err }

			// v = beta2 * v + (1 - beta2) * grad^2
			o.second_moments[i].napply([v.grad], fn [o] [T](vals []T, idx []int) T {
				g := f64(vals[1])
				return vtl.cast[T](o.beta2 * f64(vals[0]) + (1.0 - o.beta2) * g * g)
			}) or { return err }

			// theta = theta - lr_t * m / (sqrt(v) + eps)
			m_i := o.first_moments[i]
			v_i := o.second_moments[i]
			v.value.napply([m_i, v_i], fn [o, lr_t] [T](vals []T, idx []int) T {
				theta := f64(vals[0])
				m := f64(vals[1])
				vv := f64(vals[2])
				return vtl.cast[T](theta - lr_t * m / (math.sqrt(vv) + o.epsilon))
			}) or { return err }

			v.grad = vtl.zeros_like[T](v.value)
		}
	}
}
