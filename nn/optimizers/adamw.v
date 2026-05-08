module optimizers

import math
import vtl.autograd
import vtl.nn.types
import vtl

pub struct AdamWOptimizer[T] {
	learning_rate f64
	epsilon       f64
pub mut:
	beta1          f64
	beta2          f64
	beta1_t        f64
	beta2_t        f64
	weight_decay   f64
	params         []&autograd.Variable[T]
	first_moments  []&vtl.Tensor[T]
	second_moments []&vtl.Tensor[T]
}

@[params]
pub struct AdamWOptimizerConfig {
	learning_rate f64 = 0.001
	beta1         f64 = 0.9
	beta2         f64 = 0.999
	epsilon       f64 = 1e-8
	weight_decay  f64 = 0.01
}

pub fn adamw[T](config AdamWOptimizerConfig) &AdamWOptimizer[T] {
	return &AdamWOptimizer[T]{
		learning_rate: config.learning_rate
		beta1:         config.beta1
		beta2:         config.beta2
		epsilon:       config.epsilon
		weight_decay:  config.weight_decay
		beta1_t:       config.beta1
		beta2_t:       config.beta2
	}
}

pub fn (mut o AdamWOptimizer[T]) build_params(layers []types.Layer[T]) {
	for layer in layers {
		for v in layer.variables() {
			o.params << v
			o.first_moments << vtl.zeros_like[T](v.grad)
			o.second_moments << vtl.zeros_like[T](v.grad)
		}
	}
}

pub fn (mut o AdamWOptimizer[T]) update() ! {
	lr_t := o.learning_rate * math.sqrt(1.0 - o.beta2_t) / (1.0 - o.beta1_t)

	o.beta1_t *= o.beta1
	o.beta2_t *= o.beta2

	for i, mut v in o.params {
		if v.requires_grad {
			o.first_moments[i].napply([v.grad], fn [o] [T](vals []T, idx []int) T {
				return vtl.cast[T](o.beta1 * f64(vals[1]) + (1.0 - o.beta1) * f64(vals[0]))
			}) or { return err }

			o.second_moments[i].napply([v.grad], fn [o] [T](vals []T, idx []int) T {
				grad := f64(vals[0])
				return vtl.cast[T](o.beta2 * f64(vals[1]) + (1.0 - o.beta2) * grad * grad)
			}) or { return err }

			v.value.napply([o.first_moments[i], o.second_moments[i]], fn [o, lr_t] [T](vals []T, idx []int) T {
				theta := f64(vals[0])
				m := f64(vals[1])
				v_ := f64(vals[2])
				wd := o.weight_decay
				return vtl.cast[T](theta - lr_t * (m / (math.sqrt(v_) + o.epsilon) + wd * theta))
			}) or { return err }

			v.grad = vtl.zeros_like[T](v.value)
		}
	}
}
