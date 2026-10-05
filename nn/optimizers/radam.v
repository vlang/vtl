module optimizers

import vtl
import vtl.autograd
import vtl.nn.types

// RAdamOptimizer implements Rectified Adam with adaptive variance rectification.
// Rectification threshold and epsilon placement follow the PyTorch RAdam algorithm.
pub struct RAdamOptimizer[T] {
	learning_rate          f64
	epsilon                f64
	weight_decay           f64
	decoupled_weight_decay bool
pub mut:
	beta1          f64
	beta2          f64
	beta1_t        f64
	beta2_t        f64
	step           int
	params         []&autograd.Variable[T]
	first_moments  []&vtl.Tensor[T]
	second_moments []&vtl.Tensor[T]
}

// RAdamOptimizerConfig configures RAdamOptimizer.
@[params]
pub struct RAdamOptimizerConfig {
pub:
	learning_rate          f64 = 0.001
	beta1                  f64 = 0.9
	beta2                  f64 = 0.999
	epsilon                f64 = 1e-8
	weight_decay           f64
	decoupled_weight_decay bool
}

// radam_optimizer creates a RAdam optimizer.
pub fn radam_optimizer[T](config RAdamOptimizerConfig) &RAdamOptimizer[T] {
	return &RAdamOptimizer[T]{
		learning_rate:          config.learning_rate
		beta1:                  config.beta1
		beta2:                  config.beta2
		epsilon:                config.epsilon
		weight_decay:           config.weight_decay
		decoupled_weight_decay: config.decoupled_weight_decay
		beta1_t:                1.0
		beta2_t:                1.0
	}
}

// build_params registers trainable variables and initializes moment estimates.
pub fn (mut o RAdamOptimizer[T]) build_params(layers []types.Layer[T]) {
	for layer in layers {
		for v in layer.variables() {
			o.params << v
			o.first_moments << vtl.zeros_like[T](v.grad)
			o.second_moments << vtl.zeros_like[T](v.grad)
		}
	}
}

// update performs one RAdam parameter update and zeros gradients.
pub fn (mut o RAdamOptimizer[T]) update() ! {
	o.step++
	o.beta1_t *= o.beta1
	o.beta2_t *= o.beta2
	step := RAdamStepParams{
		beta1:           o.beta1
		beta2:           o.beta2
		beta1_t:         o.beta1_t
		beta2_t:         o.beta2_t
		step:            o.step
		lr:              o.learning_rate
		epsilon:         o.epsilon
		weight_decay:    o.weight_decay
		decoupled_decay: o.decoupled_weight_decay
	}
	for i, mut parameter in o.params {
		if parameter.requires_grad {
			$if sizeof(T) == 8 {
				grad := parameter.grad.to_array()
				mut theta := parameter.value.to_array()
				mut m := o.first_moments[i].to_array()
				mut v := o.second_moments[i].to_array()
				radam_step_f64_cpu(grad, mut theta, mut m, mut v, step)
				parameter.value = vtl.from_array(theta, parameter.value.shape) or { return err }
				o.first_moments[i] = vtl.from_array(m, parameter.value.shape) or { return err }
				o.second_moments[i] = vtl.from_array(v, parameter.value.shape) or { return err }
			} $else $if sizeof(T) == 4 {
				grad := parameter.grad.to_array()
				mut theta := parameter.value.to_array()
				mut m := o.first_moments[i].to_array()
				mut v := o.second_moments[i].to_array()
				radam_step_f32_cpu(grad, mut theta, mut m, mut v, step)
				parameter.value = vtl.from_array(theta, parameter.value.shape) or { return err }
				o.first_moments[i] = vtl.from_array(m, parameter.value.shape) or { return err }
				o.second_moments[i] = vtl.from_array(v, parameter.value.shape) or { return err }
			} $else {
				return error('RAdamOptimizer.update: unsupported element type size ${sizeof(T)}')
			}
			parameter.grad = vtl.zeros_like[T](parameter.value)
		}
	}
}
