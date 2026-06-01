module optimizers

import math
import vtl
import vtl.autograd
import vtl.nn.types

// AdamOptimizer implements the Adam optimiser (Adaptive Moment Estimation).
//
// Maintains per-parameter first-moment (mean) and second-moment (uncentred variance)
// moving averages of the gradients, with bias correction applied at each step.
//
// Update rule:
//   m = β₁·m + (1-β₁)·g
//   v = β₂·v + (1-β₂)·g²
//   θ = θ - lr · √(1-β₂ᵗ) / (1-β₁ᵗ) · m / (√v + ε)
//
// Reference: Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2014).
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

// AdamOptimizerConfig configures AdamOptimizer.
//
// Fields:
//   - `learning_rate` — step size α (default: 0.001)
//   - `beta1`         — exponential decay rate for first moment estimates (default: 0.9)
//   - `beta2`         — exponential decay rate for second moment estimates (default: 0.999)
//   - `epsilon`       — small constant for numerical stability (default: 1e-8)

// AdamOptimizerConfig defines a public data structure for this module.

// AdamOptimizerConfig defines a public data structure for this module.
@[params]
pub struct AdamOptimizerConfig {
pub:
	learning_rate f64 = 0.001
	beta1         f64 = 0.9
	beta2         f64 = 0.999
	epsilon       f64 = 1e-8
}

// adam_optimizer creates a new AdamOptimizer with the given configuration.
//
// Example:
// ```v
// import vtl.nn.optimizers
// opt := optimizers.adam_optimizer[f64](learning_rate: 0.001)
// opt.build_params(model.layers())
// // inside training loop:
// opt.update()!
// ```
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

// build_params registers all trainable variables from `layers` into the optimizer.
// Call once after constructing the model, before the first `update()`.
pub fn (mut o AdamOptimizer[T]) build_params(layers []types.Layer[T]) {
	for layer in layers {
		for v in layer.variables() {
			o.params << v
			o.first_moments << vtl.zeros_like[T](v.grad)
			o.second_moments << vtl.zeros_like[T](v.grad)
		}
	}
}

// update performs one Adam parameter update step and zeros all gradients.
// Must be called after `loss.backward()`.
pub fn (mut o AdamOptimizer[T]) update() ! {
	lr_t := o.learning_rate * math.sqrt(1.0 - o.beta2_t) / (1.0 - o.beta1_t)

	o.beta1_t *= o.beta1
	o.beta2_t *= o.beta2

	step := AdamStepParams{
		beta1:   o.beta1
		beta2:   o.beta2
		lr_t:    lr_t
		epsilon: o.epsilon
	}
	for i, mut v in o.params {
		if v.requires_grad {
			$if sizeof(T) == 8 {
				$if cuda ? {
					if adam_use_cuda_optimizer() {
						adam_update_f64_cuda(voidptr(v), voidptr(o.first_moments[i]),
							voidptr(o.second_moments[i]), step, i) or { return err }
						v.grad = vtl.zeros_like[T](v.value)
						continue
					}
				}
				grad := v.grad.to_array()
				mut theta := v.value.to_array()
				mut m_arr := o.first_moments[i].to_array()
				mut v_arr := o.second_moments[i].to_array()
				adam_step_f64_cpu(grad, mut theta, mut m_arr, mut v_arr, step)
				v.value = vtl.from_array(theta, v.value.shape) or { return err }
				o.first_moments[i] = vtl.from_array(m_arr, v.value.shape) or { return err }
				o.second_moments[i] = vtl.from_array(v_arr, v.value.shape) or { return err }
			} $else $if sizeof(T) == 4 {
				$if vulkan ? {
					if adam_use_vulkan_optimizer() {
						mut m_t := o.first_moments[i]
						mut v_t := o.second_moments[i]
						if try_adam_update_f32_vulkan(mut v, mut m_t, mut v_t, step) {
							o.first_moments[i] = m_t
							o.second_moments[i] = v_t
							v.grad = vtl.zeros_like[T](v.value)
							continue
						}
					}
				}
				grad := v.grad.to_array()
				mut theta := v.value.to_array()
				mut m_arr := o.first_moments[i].to_array()
				mut v_arr := o.second_moments[i].to_array()
				adam_step_f32_cpu(grad, mut theta, mut m_arr, mut v_arr, step)
				v.value = vtl.from_array(theta, v.value.shape) or { return err }
				o.first_moments[i] = vtl.from_array(m_arr, v.value.shape) or { return err }
				o.second_moments[i] = vtl.from_array(v_arr, v.value.shape) or { return err }
			} $else {
				return error('AdamOptimizer.update: unsupported element type size ${sizeof(T)}')
			}
			v.grad = vtl.zeros_like[T](v.value)
		}
	}
}
