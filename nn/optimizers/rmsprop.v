module optimizers

import math
import vtl.autograd
import vtl.nn.types
import vtl

// RMSPropOptimizer uses RMSProp with optional momentum.
// Uses a moving average of squared gradients (second moment).
pub struct RMSPropOptimizer[T] {
	learning_rate f64
	epsilon       f64
pub mut:
	alpha         f64   // smoothing constant
	momentum      f64   // momentum factor (0 = no momentum)
	weight_decay  f64
	params        []&autograd.Variable[T]
	second_moments []&vtl.Tensor[T]
	momentums     []&vtl.Tensor[T]
}

@[params]
pub struct RMSPropOptimizerConfig {
	learning_rate f64 = 0.001
	alpha         f64 = 0.99   // smoothing constant
	momentum      f64 = 0.0
	epsilon       f64 = 1e-8
	weight_decay  f64 = 0.0
}

pub fn rmsprop[T](config RMSPropOptimizerConfig) &RMSPropOptimizer[T] {
	return &RMSPropOptimizer[T]{
		learning_rate: config.learning_rate
		alpha:         config.alpha
		momentum:      config.momentum
		epsilon:       config.epsilon
		weight_decay:  config.weight_decay
	}
}

pub fn (mut o RMSPropOptimizer[T]) build_params(layers []types.Layer[T]) {
	for layer in layers {
		for v in layer.variables() {
			o.params << v
			o.second_moments << vtl.zeros_like[T](v.grad)
			if o.momentum > 0 {
				o.momentums << vtl.zeros_like[T](v.grad)
			}
		}
	}
}

pub fn (mut o RMSPropOptimizer[T]) update() ! {
	for i, mut v in o.params {
		if v.requires_grad {
			// Update second moment: v = alpha * v + (1-alpha) * grad^2
			// vals[0] = second_moment (being updated), vals[1] = grad
			o.second_moments[i].napply([v.grad], fn [o] [T](vals []T, idx []int) T {
				grad := f64(vals[1])
				return vtl.cast[T](o.alpha * f64(vals[0]) + (1.0 - o.alpha) * grad * grad)
			})

			if o.momentum > 0 {
				// m = momentum * m + grad / (sqrt(v) + eps)
				// vals[0] = prev momentum, vals[1] = grad
				o.momentums[i].napply([v.grad], fn [o] [T](vals []T, idx []int) T {
					grad := f64(vals[1])
					v_ := f64(o.second_moments[i])
					m_prev := f64(vals[0])
					denom := math.sqrt(v_) + o.epsilon
					return vtl.cast[T](o.momentum * m_prev + grad / denom)
				})
				// theta = theta - lr * (m + wd * theta)
				v.value.napply([o.momentums[i]], fn [o] [T](vals []T, idx []int) T {
					theta := f64(vals[0])
					m := f64(vals[1])
					return vtl.cast[T](theta - o.learning_rate * (m + o.weight_decay * theta))
				})
			} else {
				// theta = theta - lr * (grad / sqrt(v) + wd * theta)
				// vals[0] = theta, vals[1] = grad
				v.value.napply([v.grad], fn [o] [T](vals []T, idx []int) T {
					theta := f64(vals[0])
					grad := f64(vals[1])
					v_ := f64(o.second_moments[i])
					denom := math.sqrt(v_) + o.epsilon
					return vtl.cast[T](theta - o.learning_rate * (grad / denom + o.weight_decay * theta))
				})
			}

			v.grad = vtl.zeros_like[T](v.value)
		}
	}
}